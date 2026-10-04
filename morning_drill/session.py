"""
The session graph.

    prepare -> plan(progress agent) -> author(trainer agent) -> present
                                                                   |
                                     +-----------------------------+
                                     v
                              await_submission   <--- the student, via interrupt()
                                     |
                                  verify (deterministic)
                                     |
                                  grade(grader agent)
                                     |
                                  record
                                     |
                          correct? --+-- no --> back to await_submission
                                     |
                                    yes
                                     |
                                    END

One graph run is one challenge, including every retry. The retry loop really
is a loop in the graph now, because interrupt() is what waits for the student.
Choosing the NEXT challenge is the next run: the progress agent reads the
updated progress and history and briefs the trainer agent again.

Objects that cannot be serialized -- the DataFrame, the student's answer --
live on a Box held in config["configurable"], never in checkpointed state.
"""

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing import Annotated, Any, List, Optional, TypedDict

from .agents import author, decide, review, verify
from .agents.schemas import Brief, GeneratedChallenge
from .generators import generate_biomarker_dataset
from .spec import spec_to_reference_code


BOX_KEY = "box"


class Box:
    """
    The run's unserializable working set. Nodes read and write it directly.

    `seen` holds challenge ids, for excluding library challenges already
    dealt. `avoid` holds spec signatures, which is what the trainer agent is
    told not to re-ask -- a signature is stable across differently worded
    missions, and both are seeded from drill_records.json so the drill does
    not repeat itself across days. `digest` is the cross-session record.
    """

    __slots__ = (
        "dataset",
        "answer",
        "progress",
        "history",
        "seen",
        "avoid",
        "expected",
        "digest",
    )

    def __init__(self, progress=None, history=None, seen=(), avoid=(), digest=None):
        self.dataset = None
        self.answer = None
        self.expected = None
        self.progress = progress or {}
        self.history = list(history or [])
        self.seen = list(seen)
        self.avoid = list(avoid)
        self.digest = digest


def box_of(config):
    return (config or {}).get("configurable", {}).get(BOX_KEY) or Box()


class SessionState(TypedDict, total=False):
    brief: dict
    challenge: dict
    discrimination: dict
    trainer_notes: List[str]
    reference_solution: str

    attempt: int
    code: str
    answer_repr: str
    traceback: Optional[str]

    # `correct` is about the value; `passed` also requires the constraints
    # the mission stated to have been respected. A right answer reached with
    # a for-loop is correct and not a pass.
    correct: bool
    passed: bool
    violations: List[str]
    discrepancy: dict
    review: dict
    prior_hints: List[str]

    progress_note: str
    topic_status: str
    next_level: int

    # the outcome of the attempt just recorded, read by plan on the next run
    _outcome: Optional[dict]


# ----------------------------------------------------------------- nodes


def prepare(state, config):
    """
    Fresh data for the challenge about to be written. The trainer agent's
    challenge is validated against this exact frame.
    """
    box = box_of(config)
    box.dataset = generate_biomarker_dataset()
    return {}


def make_plan(model, apply_fn=None):
    def plan(state, config):
        box = box_of(config)

        outcome = state.get("_outcome")

        decision = decide(
            model,
            box.progress,
            outcome=outcome,
            history=box.history,
            avoid=box.avoid,
            digest=box.digest,
        )

        # topic_status and next_level describe the topic just FINISHED, so
        # they are persisted here, where the outcome is known -- not by the
        # record node of the challenge about to start.
        if outcome is not None and apply_fn is not None:
            apply_fn(box, outcome, decision)

        return {
            "brief": decision.brief.model_dump(),
            "progress_note": decision.note,
            "topic_status": decision.topic_status,
            "next_level": decision.next_level,
        }

    return plan


def make_author(model):
    def author_node(state, config):
        box = box_of(config)
        brief = Brief(**state["brief"])

        challenge, report, notes = author(
            model, brief, box.dataset, seen=box.seen
        )

        box.seen.append(challenge.id)
        box.avoid.append(challenge.spec_signature)

        return {
            "challenge": challenge.model_dump(),
            "discrimination": report,
            "trainer_notes": notes,
            "reference_solution": spec_to_reference_code(challenge.spec),
            "attempt": 0,
            "prior_hints": [],
        }

    return author_node


def await_submission(state, config):
    """
    The student. Everything above this line happened without them.
    """
    challenge = state["challenge"]

    submission = interrupt(
        {
            "kind": "submission",
            "challenge_id": GeneratedChallenge(**challenge).id,
            "title": challenge["title"],
            "mission": challenge["mission"],
            "attempt": state.get("attempt", 0) + 1,
        }
    )

    return {
        "code": (submission or {}).get("code", ""),
        "traceback": (submission or {}).get("traceback"),
        "attempt": state.get("attempt", 0) + 1,
    }


def verify_node(state, config):
    box = box_of(config)
    challenge = GeneratedChallenge(**state["challenge"])

    verification = verify(
        challenge,
        box.dataset,
        box.answer,
        state.get("code", ""),
    )

    box.expected = verification["expected"]

    return {
        "correct": verification["correct"],
        "passed": verification["correct"] and not verification["violations"],
        "violations": verification["violations"],
        "discrepancy": verification["discrepancy"],
        "answer_repr": _safe_repr(box.answer),
    }


def make_grade(model):
    def grade(state, config):
        box = box_of(config)
        challenge = GeneratedChallenge(**state["challenge"])

        verification = {
            "correct": state["correct"],
            "expected": box.expected,
            "violations": state["violations"],
            "facts": _facts_for(state),
            "discrepancy": state.get("discrepancy", {}),
        }

        result = review(
            model,
            challenge,
            state.get("code", ""),
            state.get("answer_repr", ""),
            verification,
            state.get("attempt", 1),
            prior_hints=state.get("prior_hints", []),
            traceback_text=state.get("traceback"),
        )

        update = {"review": result.model_dump()}

        if result.hint:
            update["prior_hints"] = state.get("prior_hints", []) + [result.hint]

        return update

    return grade


def make_record(record_fn):
    def record(state, config):
        box = box_of(config)
        challenge = state["challenge"]

        entry = {
            "topic": challenge["topic"],
            "level": challenge["level"],
            "passed": bool(state.get("passed")),
            "correct": bool(state.get("correct")),
            "attempt": state.get("attempt", 1),
            "challenge_id": GeneratedChallenge(**challenge).id,
            "root_cause": state.get("review", {}).get("root_cause", ""),
        }

        if state.get("passed"):
            box.history.append(entry)

        if record_fn is not None:
            record_fn(box, entry, state)

        return {"_outcome": entry}

    return record


def route_after_record(state):
    return END if state.get("passed") else "await_submission"


# ------------------------------------------------------------------ helpers


def _safe_repr(obj, limit=400):
    try:
        if isinstance(obj, pd.DataFrame):
            return f"DataFrame{obj.shape}\n{obj.head(5)}"[:limit]
        return repr(obj)[:limit]
    except Exception as exc:
        return f"(unrepresentable: {exc})"


class _Facts:
    """
    Rehydrated static facts, so the grader prompt reads the same whether the
    graph ran straight through or resumed from a checkpoint.
    """

    def __init__(self, code):
        from .constraints import analyse_code

        self.inner = analyse_code(code)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __repr__(self):
        return repr(self.inner)


def _facts_for(state):
    return _Facts(state.get("code", ""))


# -------------------------------------------------------------------- graph


def build_session_graph(
    models=None,
    record_fn=None,
    apply_fn=None,
    checkpointer=None,
):
    """
    models: {"progress": m, "trainer": m, "grader": m} or None for no agents
    """
    models = models or {}

    builder = StateGraph(SessionState)

    builder.add_node("prepare", prepare)
    builder.add_node("plan", make_plan(models.get("progress"), apply_fn))
    builder.add_node("author", make_author(models.get("trainer")))
    builder.add_node("await_submission", await_submission)
    builder.add_node("verify", verify_node)
    builder.add_node("grade", make_grade(models.get("grader")))
    builder.add_node("record", make_record(record_fn))

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "plan")
    builder.add_edge("plan", "author")
    builder.add_edge("author", "await_submission")
    builder.add_edge("await_submission", "verify")
    builder.add_edge("verify", "grade")
    builder.add_edge("grade", "record")

    builder.add_conditional_edges(
        "record",
        route_after_record,
        {"await_submission": "await_submission", END: END},
    )

    return builder.compile(
        checkpointer=checkpointer if checkpointer is not None else MemorySaver()
    )
