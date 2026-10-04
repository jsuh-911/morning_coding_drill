"""
Trainer agent.

Writes the challenge: prose mission for the student, declarative spec for the
machine. The two must describe the same computation, and the challenge is
rejected and sent back for revision if:

  - the spec does not validate against the dataset schema
  - the mission states the answer
  - the spec produces nothing, or something degenerate
  - the challenge is not discriminating: a student using the wrong statistic
    or dropping a filter would still pass

Nothing the agent writes is executed. The spec is interpreted by
morning_drill.spec, which is hand-written.
"""

import pandas as pd

from ..discrimination import discrimination_report
from ..spec import evaluate_spec
from . import prompts
from .library import from_library
from .schemas import GeneratedChallenge


class Rejection(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _selected_rows(df, spec):
    from ..spec import FILTER_OPS

    frame = df

    for f in spec.filters:
        frame = frame[FILTER_OPS[f.op](frame[f.column], f.value)]

    return len(frame)


def validate(challenge, df, require_full_discrimination=False):
    """
    Raise Rejection with a reason the agent can act on, or return a report.
    """
    # checked before evaluating, so the agent gets "your filters select
    # nothing" rather than an argmax-of-empty-sequence traceback
    if _selected_rows(df, challenge.spec) == 0:
        raise Rejection(
            "the filters select no rows, so the question has no answer; "
            "check for conditions that cannot hold at the same time"
        )

    try:
        answer = evaluate_spec(df, challenge.spec)
    except Exception as exc:
        raise Rejection(f"the spec cannot be evaluated: {exc}")

    if isinstance(answer, pd.Series):
        if answer.empty:
            raise Rejection("the spec selects no rows, so there is no answer")
        if answer.isna().all():
            raise Rejection("every group aggregates to NaN")
    else:
        if answer is None or (isinstance(answer, float) and pd.isna(answer)):
            raise Rejection("the spec aggregates to NaN")

    mission = challenge.mission.lower()

    # A value the spec filters on may legitimately appear in the mission:
    # "among NEFL measurements..." is a premise, not a leak. Only an answer
    # the student is supposed to find counts.
    filter_values = set()

    for f in challenge.spec.filters:
        values = f.value if isinstance(f.value, list) else [f.value]
        filter_values.update(str(v).lower() for v in values if v is not None)

    if (
        isinstance(answer, str)
        and answer.lower() in mission
        and answer.lower() not in filter_values
    ):
        raise Rejection(
            f"the mission contains the answer ({answer!r}); state what to "
            "compute, not the result"
        )

    if not challenge.mission.strip():
        raise Rejection("the mission is empty")

    if challenge.level >= 3 and challenge.toolbox:
        raise Rejection(
            f"level {challenge.level} must not supply a toolbox"
        )

    single_valued = challenge.spec.reduce not in ("none", "nlargest", "nsmallest")
    two_output_phrases = ("report both", "along with its", "as well as its",
                          "and its value", "and the value")

    if single_valued and any(p in mission for p in two_output_phrases):
        raise Rejection(
            f"the spec returns one {challenge.spec.answer_kind} but the "
            "mission asks for two things; either ask only for the one value, "
            "or change reduce to `none` so a Series is returned"
        )

    report = discrimination_report(df, challenge.spec)

    if not report["discriminating"]:
        raise Rejection(
            "this challenge is not discriminating: a student who used the "
            "wrong statistic or dropped a filter would get the same answer "
            "and pass with the wrong idea. Make the computation depend on the "
            "choice of statistic, for example by aggregating a skewed column "
            "or by requiring a filter that changes which group wins."
        )

    if require_full_discrimination and report["missed"]:
        raise Rejection(
            "these mistakes would still pass: "
            f"{report['missed']}. Tighten the question so each one changes "
            "the answer."
        )

    return report


def author(model, brief, df, max_revisions=2, seen=()):
    """
    Trainer agent -> (GeneratedChallenge, discrimination report, notes).

    With no model, or after too many rejections, falls back to the library.
    """
    notes = []

    if model is None:
        challenge = from_library(brief.topic, brief.level, exclude=seen)
        return challenge, discrimination_report(df, challenge.spec), notes

    structured = model.with_structured_output(GeneratedChallenge)

    prompt = prompts.TRAINER_PROMPT.format(
        rules=prompts.SHARED_RULES,
        dataset=prompts.DATASET_BRIEF,
        domain=brief.domain,
        topic=brief.topic,
        level=brief.level,
        rationale=brief.rationale,
        fold_in=", ".join(brief.fold_in) or "nothing",
        avoid=", ".join(brief.avoid) or "nothing",
    )

    challenge = None

    for attempt in range(max_revisions + 1):
        try:
            candidate = structured.invoke(prompt)
        except Exception as exc:
            notes.append(f"trainer agent unavailable ({exc}); used the library")
            challenge = from_library(brief.topic, brief.level, exclude=seen)
            return challenge, discrimination_report(df, challenge.spec), notes

        try:
            report = validate(
                candidate,
                df,
                require_full_discrimination=(attempt == 0),
            )
            return candidate, report, notes
        except Rejection as rejection:
            notes.append(f"revision {attempt + 1}: {rejection.reason}")

            prompt = (
                prompts.TRAINER_PROMPT.format(
                    rules=prompts.SHARED_RULES,
                    dataset=prompts.DATASET_BRIEF,
                    domain=brief.domain,
                    topic=brief.topic,
                    level=brief.level,
                    rationale=brief.rationale,
                    fold_in=", ".join(brief.fold_in) or "nothing",
                    avoid=", ".join(brief.avoid) or "nothing",
                )
                + "\n\n"
                + prompts.REVISION_PROMPT.format(
                    mission=candidate.mission,
                    spec=candidate.spec.model_dump_json(),
                    problem=rejection.reason,
                )
            )
            challenge = candidate

    notes.append("the trainer agent could not produce a sound challenge")
    fallback = from_library(brief.topic, brief.level, exclude=seen)
    return fallback, discrimination_report(df, fallback.spec), notes
