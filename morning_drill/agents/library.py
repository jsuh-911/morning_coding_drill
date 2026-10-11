"""
Hand-written challenges, used when no model is available.

The trainer agent generates challenges at runtime; these let the whole system
run offline, and they are what the fallback brief draws on.
"""

from ..spec import ChallengeSpec
from .schemas import GeneratedChallenge

LIBRARY = [
    GeneratedChallenge(
        title="Highest GFAP group",
        mission=(
            "Among samples passing QC (qc_score >= 0.85), identify which "
            "disease group has the highest mean GFAP intensity."
        ),
        toolbox=["boolean filtering", "groupby", "mean", "idxmax"],
        constraint_ids=["no_loops", "use_pandas"],
        spec=ChallengeSpec(
            filters=[
                {"column": "qc_score", "op": ">=", "value": 0.85},
                {"column": "protein", "op": "==", "value": "GFAP"},
            ],
            group_by=["disease"],
            value="intensity",
            agg="mean",
            reduce="idxmax",
        ),
        topic="groupby",
        level=1,
    ),
    GeneratedChallenge(
        title="Spread of GFAP across groups",
        mission=(
            "Using only QC-passing measurements (qc_score >= 0.85), report the difference between "
            "the largest and smallest group-level median GFAP intensity across "
            "disease groups."
        ),
        toolbox=["boolean filtering", "groupby", "median"],
        constraint_ids=["no_loops", "use_pandas"],
        spec=ChallengeSpec(
            filters=[
                {"column": "qc_score", "op": ">=", "value": 0.85},
                {"column": "protein", "op": "==", "value": "GFAP"},
            ],
            group_by=["disease"],
            value="intensity",
            agg="median",
            reduce="diff_extremes",
        ),
        topic="groupby",
        level=2,
    ),
    GeneratedChallenge(
        title="Three most variable proteins in AD",
        mission=(
            "Within the AD cohort, restricted to QC-passing measurements (qc_score >= 0.85), "
            "return the three proteins with the largest standard deviation of "
            "intensity, as a Series ordered from largest to smallest."
        ),
        toolbox=[],
        constraint_ids=["no_loops", "use_pandas", "no_apply"],
        spec=ChallengeSpec(
            filters=[
                {"column": "disease", "op": "==", "value": "AD"},
                {"column": "qc_score", "op": ">=", "value": 0.85},
            ],
            group_by=["protein"],
            value="intensity",
            agg="std",
            reduce="nlargest",
            n=3,
        ),
        topic="groupby",
        level=3,
    ),
    GeneratedChallenge(
        title="Extreme reading by assay",
        mission=(
            "Restricting to QC-passing measurements (qc_score >= 0.85), determine which protein "
            "assay produced the single highest intensity reading in the study."
        ),
        toolbox=[],
        constraint_ids=["no_loops", "use_pandas"],
        spec=ChallengeSpec(
            filters=[{"column": "qc_score", "op": ">=", "value": 0.85}],
            group_by=["protein"],
            value="intensity",
            agg="max",
            reduce="idxmax",
        ),
        topic="groupby",
        level=4,
    ),
]


def from_library(topic, level, exclude=(), never=()):
    """
    Closest library challenge to a brief.

    `exclude` is what has been dealt already; when it empties the pool the
    library is recycled. `never` is what the learner flagged as wrong, and
    stays out even then.
    """
    pool = [c for c in LIBRARY if c.id not in exclude and c.id not in never]

    if not pool:
        pool = [c for c in LIBRARY if c.id not in never]

    if not pool:
        raise RuntimeError(
            "every library question has been flagged as wrong, so there is "
            "nothing left to deal without a model. Fix the library or turn "
            "the agents on."
        )

    on_topic = [c for c in pool if c.topic == topic] or pool

    return min(on_topic, key=lambda c: abs(c.level - level))
