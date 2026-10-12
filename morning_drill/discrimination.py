"""
Is a challenge actually testing what it claims?

The drill's original groupby challenge asked for a median, but the synthetic
AD/NEFL signal was so strong that mean, max and median all gave the same
answer. A learner could hold the wrong idea and still pass.

Before a generated challenge is handed out, perturb its spec the way a
confused student would and require that at least one perturbation changes the
answer. A challenge that survives every perturbation is not discriminating,
and the trainer agent is asked to revise it.
"""

import pandas as pd

from .spec import evaluate_spec

SWAPS = {
    "median": "mean",
    "mean": "median",
    "max": "min",
    "min": "max",
    "sum": "mean",
    "count": "nunique",
    "std": "mean",
    "var": "mean",
    "range": "std",
    "nunique": "count",
}


TRIVIAL = {"idxmin instead of idxmax"}


def _same(a, b):
    try:
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            return a.equals(b)
        if isinstance(a, pd.Series) or isinstance(b, pd.Series):
            return False
        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b) < 1e-9
        return bool(a == b)
    except Exception:
        return False


def perturbations(spec):
    """
    Plausible wrong versions of the same spec.
    """
    out = []

    swapped = SWAPS.get(spec.agg)

    if swapped:
        out.append(("wrong statistic", spec.model_copy(update={"agg": swapped})))

    for i, f in enumerate(spec.filters):
        remaining = [g for j, g in enumerate(spec.filters) if j != i]
        out.append(
            (
                f"filter on {f.column} dropped",
                spec.model_copy(update={"filters": remaining}),
            )
        )

    if spec.reduce == "idxmax":
        out.append(
            ("idxmin instead of idxmax", spec.model_copy(update={"reduce": "idxmin"}))
        )

    return out


def discrimination_report(df, spec):
    """
    Which student mistakes this challenge would actually catch.
    """
    truth = evaluate_spec(df, spec)

    caught = []
    missed = []

    for label, variant in perturbations(spec):
        try:
            result = evaluate_spec(df, variant)
        except Exception:
            continue

        (missed if _same(truth, result) else caught).append(label)

    # Flipping idxmax to idxmin changes the answer of every idxmax challenge,
    # so catching it proves nothing about the statistic or the filters. Only
    # a mistake about the pipeline itself counts toward "discriminating".
    substantive = [label for label in caught if label not in TRIVIAL]

    return {
        "caught": caught,
        "missed": missed,
        "discriminating": bool(substantive),
        "summary": (
            f"catches {len(caught)} of {len(caught) + len(missed)} "
            "plausible mistakes"
        ),
    }
