"""
How often does a random question tell right from wrong?

Draw random specs the way the trainer agent might and report the share that
are discriminating (some wrong statistic or dropped filter changes the answer)
and the share where EVERY plausible mistake does. The number to move when the
dataset changes.

    python -m scripts.discrimination_yield [n_specs] [version]
"""

import random
import sys

from morning_drill.discrimination import discrimination_report
from morning_drill.generators import generate_biomarker_dataset
from morning_drill.spec import ChallengeSpec, Filter

AGGS = ["median", "mean", "max", "min", "sum", "std"]
CUTOFFS = [0.6, 0.7, 0.8, 0.85, 0.9]


def random_spec(rng):
    group = rng.choice([["disease"], ["protein"]])
    filters = []

    if rng.random() < 0.8:
        filters.append(Filter(column="qc_score", op=">=", value=rng.choice(CUTOFFS)))

    if group == ["disease"] and rng.random() < 0.7:
        filters.append(
            Filter(
                column="protein",
                op="==",
                value=rng.choice(["MAPT", "SNCA", "NEFL", "GFAP", "TREM2"]),
            )
        )

    if group == ["protein"] and rng.random() < 0.5:
        filters.append(
            Filter(column="disease", op="==", value=rng.choice(["Control", "AD", "PD", "ALS"]))
        )

    return ChallengeSpec(
        filters=filters,
        group_by=group,
        value="intensity",
        agg=rng.choice(AGGS),
        reduce=rng.choice(["idxmax", "idxmin"]),
    )


def measure(n_specs=150, version=2, n_rows=20_000, seed=0):
    rng = random.Random(seed)
    df = generate_biomarker_dataset(n=n_rows, seed=3, version=version)

    discriminating = full = 0

    for _ in range(n_specs):
        report = discrimination_report(df, random_spec(rng))
        discriminating += report["discriminating"]
        full += report["discriminating"] and not report["missed"]

    return {
        "specs": n_specs,
        "discriminating": round(discriminating / n_specs, 2),
        "fully_discriminating": round(full / n_specs, 2),
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    versions = [int(sys.argv[2])] if len(sys.argv) > 2 else [1, 2]

    for v in versions:
        print(f"v{v}", measure(n, v))
