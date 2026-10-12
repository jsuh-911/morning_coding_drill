"""
Grader agent.

Reviews the submitted CODE: constraint compliance, soundness of the
reasoning, idiom, and what the student did well. It is told whether the
ANSWER was right, and cannot overturn that — correctness is computed from the
challenge spec by morning_drill.spec, not judged.
"""

from ..constraints import CONSTRAINT_CHECKS, check_constraints, looks_hardcoded
from ..graders import compare_answers, discrepancy_report
from ..probes import method_failures
from ..spec import evaluate_spec
from . import prompts
from .schemas import Review


def verify(challenge, dataset, answer, code, runner=None, prepare=None):
    """
    The deterministic half. No model involved.

    `dataset` is the clean table the expected answer is computed on. On a
    cleaning topic the student was handed a raw copy; `prepare` rebuilds that
    raw copy for any frame, so probes run on what they would have seen.
    """
    expected = evaluate_spec(dataset, challenge.spec)
    correct = compare_answers(answer, expected)

    violations, facts = check_constraints(code, challenge.constraint_ids)

    hardcoded = correct and looks_hardcoded(facts, expected)

    if hardcoded:
        correct = False
        violations.append(
            "the answer was written in as a literal rather than computed"
        )

    # A right value is not yet a right method: on this data several wrong
    # pipelines land on the same label. Run the code on data built to tell
    # them apart.
    method_failed = False

    if correct and not hardcoded:
        failures = method_failures(
            code, dataset, challenge.spec, runner, prepare
        )

        if failures:
            correct = False
            method_failed = True
            violations.extend(failures)

    # a hardcoded answer has no structural discrepancy to report: the value
    # is right, the method is the problem
    discrepancy = (
        {} if correct or hardcoded or method_failed else discrepancy_report(answer, expected)
    )

    return {
        "correct": bool(correct),
        "expected": expected,
        "violations": violations,
        "facts": facts,
        "discrepancy": discrepancy,
        "hardcoded": bool(hardcoded),
    }


def _constraint_text(constraint_ids):
    """
    The wording the student was shown, not the internal ids -- otherwise the
    grader agent quotes `no_loops` back at them.
    """
    if not constraint_ids:
        return "none"

    return "\n".join(
        f"  - {CONSTRAINT_CHECKS.get(cid, (cid, None))[0]}" for cid in constraint_ids
    )


def fallback_review(verification, attempt):
    """
    Deterministic review, used when no model is available.
    """
    if verification["correct"] and not verification["violations"]:
        return Review(
            error_class="correct",
            root_cause="",
            reasoning_sound=True,
            strengths="Correct.",
        )

    facts = verification["facts"]

    if facts.syntax_error:
        error_class = "syntax"
    elif verification["violations"]:
        error_class = "style_only" if verification["correct"] else "conceptual"
    else:
        notes = verification["discrepancy"].get("notes", [])
        error_class = (
            "wrong_shape"
            if any("expected" in n and "submitted" in n for n in notes)
            else "conceptual"
        )

    hint = " ".join(verification["discrepancy"].get("notes", []))

    if verification.get("hardcoded"):
        return Review(
            error_class="conceptual",
            root_cause=(
                "the value is right but it was typed in rather than derived "
                "from the data; the drill is the derivation"
            ),
            reasoning_sound=False,
            hint="Compute it from df instead of stating it.",
        )

    if verification["violations"]:
        hint = (
            "constraints not met: "
            + "; ".join(verification["violations"])
            + ("  " + hint if hint else "")
        )

    return Review(
        error_class=error_class,
        root_cause=hint or "the result is not what the mission asked for",
        reasoning_sound=False,
        hint=hint,
    )


def review(
    model,
    challenge,
    code,
    answer_repr,
    verification,
    attempt,
    prior_hints=(),
    traceback_text=None,
):
    """
    Grader agent -> Review.
    """
    if model is None:
        return fallback_review(verification, attempt)

    task = prompts.TASK_BY_ATTEMPT[
        0 if verification["correct"] else min(attempt, 3)
    ]

    structural = ""

    if verification["discrepancy"]:
        notes = verification["discrepancy"]["notes"]
        structural = "  structural report:     " + "; ".join(notes)

    prior = (
        "\n".join(f"  - {h}" for h in prior_hints)
        if prior_hints
        else "  no hints given yet"
    )

    prompt = prompts.GRADER_PROMPT.format(
        rules=prompts.SHARED_RULES,
        dataset=prompts.DATASET_BRIEF,
        mission=challenge.mission,
        constraints=_constraint_text(challenge.constraint_ids),
        code=code or "(not captured)",
        answer_repr=answer_repr,
        traceback_section=(
            f"IT RAISED\n{traceback_text}"
            if traceback_text
            else "It ran without raising."
        ),
        correct=verification["correct"],
        violations=verification["violations"] or "none",
        facts=verification["facts"],
        structural_section=structural,
        attempt=attempt,
        prior_section=prior,
        task=task,
    )

    try:
        result = model.with_structured_output(Review).invoke(prompt)
    except Exception:
        return fallback_review(verification, attempt)

    # the agent does not get to overturn arithmetic
    if not verification["correct"] and result.error_class == "correct":
        result.error_class = "conceptual"

    if verification["correct"] and verification["violations"]:
        result.error_class = "style_only"

    return result
