"""
Blind solver.

A mission is prose and the spec is what decides correctness, and nothing
forces the two to agree. Pattern-matching the prose for known mistakes only
catches the mistakes already thought of.

So the check is a round trip. A model reads the mission alone, with the
intended spec withheld, and writes its own spec. Both specs are evaluated on
the real data. If a careful reader of the mission gets a different answer, or
a different kind of answer, from the one the grader will accept, then a
student reading it the same way would be marked wrong: the mission is
unanswerable or ambiguous, whichever words caused it.

This needs a model, so it only runs when the trainer agent does. Library
challenges are hand-written and tested.
"""

from ..graders import compare_answers
from ..spec import evaluate_spec
from . import prompts
from .schemas import AnswerForm, BlindSolution

SPEC_FIELDS = ("filters", "group_by", "value", "agg", "reduce", "n")


def _describe(spec):
    parts = [
        "filter " + ", ".join(
            f"{f.column} {f.op} {f.value!r}" if f.value is not None
            else f"{f.column} {f.op}"
            for f in spec.filters
        )
        if spec.filters
        else "no filter",
        f"group by {', '.join(spec.group_by)}",
        f"{spec.agg} of {spec.value}",
        f"reduce {spec.reduce}",
    ]

    return "; ".join(parts)


def _differences(intended, read):
    out = []

    for name in SPEC_FIELDS:
        if getattr(intended, name) != getattr(read, name):
            out.append(name)

    return out


def solve_blind(model, mission, samples=2):
    """
    `samples` independent readings of the mission. A failed call is dropped
    rather than raised: a flaky solver must not block good challenges.
    """
    structured = model.with_structured_output(BlindSolution)
    prompt = prompts.SOLVER_PROMPT.format(
        dataset=prompts.DATASET_BRIEF, mission=mission
    )

    readings = []

    for _ in range(samples):
        try:
            readings.append(structured.invoke(prompt))
        except Exception:
            continue

    return readings


def read_answer_form(model, mission, samples=2):
    """
    What a reader of the mission alone takes the final answer to be.

    Asked WITHOUT the spec grammar. Shown the reducers, a reader maps
    "spread" onto diff_extremes and calls the question fine; asked only what
    the answer is, "which protein" is a name and cannot be a number.
    """
    structured = model.with_structured_output(AnswerForm)
    prompt = prompts.FORM_PROMPT.format(mission=mission)

    forms = []

    for _ in range(samples):
        try:
            forms.append(structured.invoke(prompt))
        except Exception:
            continue

    return forms


def check_mission(model, challenge, df, samples=2):
    """
    (reason, status): reason is None when the mission holds up.

    status is "agreed", "skipped" (no usable reading, so nothing was proved
    either way) or "rejected".
    """
    # Step 1: the shape of the answer, from the words alone.
    # The student sees the title as well; a title that asks something else
    # ("Which group...") over a mission that asks for a number is a conflict.
    forms = read_answer_form(
        model, f"{challenge.title}\n{challenge.mission}", samples
    )
    wanted = challenge.spec.answer_form

    for form in forms:
        if not form.clear:
            return (
                "a reader of the mission alone found it unclear what the "
                f"answer is: {form.problem or 'no reason given'}. Ask for "
                "exactly one thing.",
                "rejected",
            )

        if form.asks_for != wanted:
            return (
                f"a reader of the mission alone takes the answer to be a "
                f"{form.asks_for}, but the spec returns a {wanted} "
                f"(reduce={challenge.spec.reduce}). Make the prose ask for a "
                f"{wanted}, or change the spec's reduce.",
                "rejected",
            )

    # Step 2: the computation, from the words alone.
    readings = solve_blind(model, challenge.mission, samples)

    if not readings:
        return None, "skipped" if not forms else "agreed"

    intended = evaluate_spec(df, challenge.spec)
    judged = 0

    for reading in readings:
        if not reading.expressible:
            return (
                "a reader of the mission alone says it cannot be answered as "
                f"worded: {reading.problem or 'no reason given'}. Reword it "
                "so it asks for exactly what the spec returns.",
                "rejected",
            )

        if reading.spec is None:
            continue

        try:
            got = evaluate_spec(df, reading.spec)
        except Exception:
            continue

        judged += 1

        if challenge.spec.answer_kind != reading.spec.answer_kind:
            return (
                f"the mission reads as asking for a {reading.spec.answer_kind} "
                f"but the spec returns a {challenge.spec.answer_kind}. "
                "Make the prose ask for what the spec returns, or change the "
                "spec's reduce.",
                "rejected",
            )

        if not compare_answers(got, intended):
            differs = _differences(challenge.spec, reading.spec)
            return (
                "a reader of the mission alone computed a different answer "
                f"({_describe(reading.spec)}) from the intended one "
                f"({_describe(challenge.spec)}); they differ in "
                f"{', '.join(differs) or 'details'}. The wording supports "
                "more than one computation. State the "
                f"{' and '.join(differs) or 'computation'} explicitly."
                + (f" Reader's note: {reading.problem}" if reading.problem else ""),
                "rejected",
            )

        if reading.ambiguous:
            return (
                "a reader of the mission alone found it ambiguous: "
                f"{reading.problem or 'more than one reasonable reading'}.",
                "rejected",
            )

    return (None, "agreed") if judged else (None, "skipped")
