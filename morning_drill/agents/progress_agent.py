"""
Progress agent.

Reviews what the student has done and decides what comes next, including
when to fold an older mastered skill back in unannounced. It speaks to the
trainer agent through a Brief.
"""

from ..curriculum import CURRICULUM, IMPLEMENTED, curriculum_text, next_topic
from ..records import digest_text, retrieval_due
from . import prompts
from .schemas import Brief, ProgressDecision


def _history_text(history):
    if not history:
        return "First challenge of the session."

    lines = []

    for item in history[-6:]:
        lines.append(
            f"  {item['topic']} L{item['level']}: "
            f"{'passed' if item['passed'] else 'failed'} "
            f"on attempt {item['attempt']}"
        )

    return "\n".join(lines)


def _outcome_text(outcome):
    if outcome is None:
        return "No challenge has been attempted yet this session."

    return (
        "THE CHALLENGE JUST FINISHED\n"
        f"  topic:   {outcome['topic']}\n"
        f"  level:   {outcome['level']}\n"
        f"  passed:  {outcome['passed']}\n"
        f"  attempts:{outcome['attempt']}\n"
        f"  review:  {outcome.get('root_cause', '')}"
    )


def _mastered_topics(progress):
    return [
        topic
        for topics in progress.values()
        for topic, state in topics.items()
        if state.get("status") == "mastered"
    ]


def fallback_decision(progress, outcome, history, digest=None):
    """
    Rule-based progression, used when no model is available.

    Encodes the same policy the prompt states, so behaviour does not change
    shape when the key is missing.
    """
    if outcome is None:
        topic, level = next_topic(progress, history)

        return ProgressDecision(
            topic_status="active",
            next_level=level,
            note="",
            brief=Brief(
                topic=topic,
                level=level,
                rationale="the active topic in the curriculum",
            ),
        )

    topic = outcome["topic"]
    level = outcome["level"]
    passed = outcome["passed"]
    attempts = outcome["attempt"]

    if passed and attempts >= 3:
        status, next_level = "needs_rework", level
        note = f"Passed, but it took {attempts} attempts. Staying at level {level}."
    elif passed and level >= 4:
        status, next_level = "mastered", 4
        note = f"{topic} survived delayed retrieval. Mastered."
    elif passed:
        status, next_level = "active", level + 1
        note = f"Level {level} cleared. Level {next_level} gives less away."
    else:
        status, next_level = "active", level
        note = ""

    # a mastered topic gone stale is due for unannounced retrieval; with no
    # records to go on, fall back to every fourth cleared challenge
    stale = (
        retrieval_due(digest, _mastered_topics(progress), exclude=(topic,))
        if digest
        else None
    )

    if stale is None:
        completed = len([h for h in history if h["passed"]])

        if completed and completed % 4 == 0:
            others = [
                name
                for name in _mastered_topics(progress)
                if name != topic
            ]
            stale = others[0] if others else None

    # Mastering a topic moves on to the next one in the curriculum. Without
    # this the brief stayed on the mastered topic forever, and the topics after
    # it were never reached.
    if status == "mastered" and not stale:
        moved = {
            domain: {name: dict(state) for name, state in topics.items()}
            for domain, topics in progress.items()
        }
        moved.setdefault(outcome.get("domain", "python"), {}).setdefault(topic, {})[
            "status"
        ] = "mastered"

        following, following_level = next_topic(moved, history)

        return ProgressDecision(
            topic_status=status,
            next_level=next_level,
            note=note,
            brief=Brief(
                topic=following,
                level=following_level,
                rationale="the next topic in the curriculum",
            ),
        )

    if stale:
        return ProgressDecision(
            topic_status=status,
            next_level=next_level,
            note=note,
            brief=Brief(
                topic=stale,
                level=4,
                rationale="delayed retrieval of an older skill",
                fold_in=[topic],
            ),
        )

    return ProgressDecision(
        topic_status=status,
        next_level=next_level,
        note=note,
        brief=Brief(
            topic=topic,
            level=next_level,
            rationale=(
                "repeat the topic that needed several attempts"
                if status == "needs_rework"
                else "continue the active topic"
            ),
        ),
    )


def decide(model, progress, outcome=None, history=None, avoid=(), digest=None):
    """
    Progress agent -> ProgressDecision.

    `digest` is the cross-session record (morning_drill.records.digest); it is
    what lets the decision depend on real elapsed time and on mistakes made
    days ago. `avoid` is spec signatures already set, so the trainer agent
    does not re-ask the same question.
    """
    history = history or []

    if model is None:
        decision = fallback_decision(progress, outcome, history, digest)
        decision.brief.avoid = list(avoid)
        return decision

    prompt = prompts.PROGRESS_PROMPT.format(
        rules=prompts.SHARED_RULES,
        curriculum=curriculum_text(),
        progress=_format_progress(progress),
        digest=digest_text(digest),
        history=_history_text(history),
        outcome_section=_outcome_text(outcome),
    )

    try:
        decision = model.with_structured_output(ProgressDecision).invoke(prompt)
    except Exception:
        decision = fallback_decision(progress, outcome, history, digest)

    decision.brief.avoid = list(avoid)

    # Only topics there is a generator for. The curriculum lists more (SQL, the
    # command line) than the trainer can write questions about; a brief for one
    # of those got a pandas question under the wrong heading, with the attempts
    # recorded against a topic that was never practised.
    chosen = (decision.brief.domain, decision.brief.topic)
    available = {
        (domain, name) for domain, names in IMPLEMENTED.items() for name in names
    }

    if chosen not in available:
        decision.brief = fallback_decision(
            progress, outcome, history, digest
        ).brief
        decision.brief.avoid = list(avoid)

    return decision


def _format_progress(progress):
    lines = []

    for domain, topics in progress.items():
        # Only what can be set. Topics seeded in progress.json for SQL and the
        # command line read as "active" to a model, which then picks one.
        settable = {
            name: state
            for name, state in topics.items()
            if name in IMPLEMENTED.get(domain, [])
        }

        if not settable:
            continue

        lines.append(f"{domain}:")

        for topic, state in settable.items():
            lines.append(
                f"  {topic}: {state.get('status')} level {state.get('level')} "
                f"attempts {state.get('attempts', 0)} "
                f"correct {state.get('correct', 0)}"
            )

    return "\n".join(lines)
