"""
The contracts between the three agents and the student.

Every agent speaks in one of these. They are pydantic models so the models
return structured output rather than prose the graph has to parse.
"""

import hashlib
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from ..spec import ChallengeSpec


DOMAINS = ("python", "sql", "cli")
LEVELS = (1, 2, 3, 4)


class Brief(BaseModel):
    """
    Progress agent -> trainer agent. What to set next, and why.
    """

    domain: Literal[DOMAINS] = "python"  # type: ignore[valid-type]
    topic: str = Field(description="curriculum topic, e.g. groupby")
    level: int = Field(ge=1, le=4, description="1 guided .. 4 delayed retrieval")
    rationale: str = Field(
        description="one sentence on why this topic at this level now"
    )
    fold_in: List[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "mastered topics to exercise again inside this challenge without "
            "announcing it; this is how delayed retrieval happens"
        ),
    )
    avoid: List[str] = Field(
        default_factory=list,
        description="spec signatures recently used, so the question is new",
    )


class GeneratedChallenge(BaseModel):
    """
    Trainer agent -> student. Prose for the human, spec for the machine.
    """

    title: str = Field(max_length=60)
    mission: str = Field(
        description=(
            "the problem, in one or two sentences, in analytical language. "
            "Must not contain the answer."
        )
    )
    toolbox: List[str] = Field(default_factory=list, max_length=6)
    constraint_ids: List[
        Literal["no_loops", "no_apply", "no_iterrows", "use_pandas"]  # type: ignore[valid-type]
    ] = Field(default_factory=list)
    spec: ChallengeSpec
    topic: str
    level: int = Field(ge=1, le=4)
    revision_note: Optional[str] = None

    @property
    def id(self):
        # hashlib, not hash(): the builtin is salted per interpreter, so ids
        # differed between runs and cross-session dedup never matched.
        fingerprint = hashlib.sha256(
            f"{self.mission}|{self.spec.signature}".encode()
        ).hexdigest()[:8]

        return f"gen_{self.topic}_{fingerprint}"

    @property
    def spec_signature(self):
        return self.spec.signature


class Review(BaseModel):
    """
    Grader agent -> student. Judgement of the CODE. The verdict on the
    ANSWER is computed, not reviewed, and is not a field here.
    """

    error_class: Literal[
        "correct",
        "syntax",
        "runtime",
        "wrong_shape",
        "conceptual",
        "style_only",
    ]
    root_cause: str = Field(
        description="one sentence; no solution, no dataset values"
    )
    reasoning_sound: bool = Field(
        description="does the pipeline express a correct line of thought"
    )
    strengths: str = Field(
        default="",
        description="specific, on a correct submission; empty otherwise",
    )
    idiomatic_note: str = Field(
        default="",
        description=(
            "a cleaner form and one sentence on why, or empty if the code "
            "is already idiomatic"
        ),
    )
    hint: str = Field(
        default="",
        description="exactly one hint, at the strength asked for",
    )


class ProgressDecision(BaseModel):
    """
    Progress agent -> progress.json + the next brief.
    """

    topic_status: Literal["active", "mastered", "needs_rework"]
    next_level: int = Field(ge=1, le=4)
    note: str = Field(description="one line for the student")
    brief: Brief
