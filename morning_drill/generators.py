import numpy as np
import pandas as pd


def generate_biomarker_dataset_v1(n=50_000, seed=42):
    """
    The original table: one planted signal (AD x NEFL), so AD wins almost any
    comparison whatever the statistic or filter. Kept for reproducing old
    results; challenges are not drawn from it.
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "sample_id": [f"S{i:05d}" for i in range(n)],
        "disease": rng.choice(
            ["Control", "AD", "PD", "ALS"],
            size=n,
            p=[0.30, 0.30, 0.25, 0.15],
        ),
        "protein": rng.choice(
            ["MAPT", "SNCA", "NEFL", "GFAP", "TREM2"],
            size=n,
        ),
        "intensity": rng.lognormal(
            mean=5,
            sigma=0.6,
            size=n,
        ),
        "qc_score": rng.uniform(
            0.5,
            1.0,
            size=n,
        ),
    })

    # Add a controlled AD-associated NEFL signal
    mask = (
        (df["disease"] == "AD")
        & (df["protein"] == "NEFL")
    )
    df.loc[mask, "intensity"] *= 1.5

    # Add some missing measurements
    missing_idx = rng.choice(
        df.index,
        size=max(1, int(n * 0.01)),
        replace=False,
    )
    df.loc[missing_idx, "intensity"] = np.nan

    return df


# ------------------------------------------------------------------- v2

DISEASES = ["Control", "AD", "PD", "ALS"]
DISEASE_P = [0.30, 0.30, 0.25, 0.15]
PROTEINS = ["MAPT", "SNCA", "NEFL", "GFAP", "TREM2"]

# (log-mean, sigma) of the baseline.
BASELINE = {
    "MAPT": (4.9, 0.45),
    "SNCA": (5.0, 0.50),
    "NEFL": (5.0, 0.50),
    "GFAP": (5.1, 0.60),
    "TREM2": (4.9, 0.50),
}

# Multiplicative disease effects. Control is mildly raised on every protein, so
# Control wins when the proteins are pooled; each protein then has a different,
# stronger winner of its own. Dropping the protein filter therefore changes the
# answer, whichever protein it was.
CONTROL_LIFT = 1.30

EFFECT = {
    ("AD", "NEFL"): 1.55,
    ("AD", "TREM2"): 1.50,
    ("PD", "SNCA"): 1.50,
    ("PD", "MAPT"): 1.45,
    ("ALS", "GFAP"): 1.50,
}

# Rare, very large readings in cells that do NOT win on the typical value. The
# median barely moves; the mean and the max do, so the wrong statistic names
# a different group. Rates are high for a real instrument and deliberate.
OUTLIERS = {
    ("PD", "NEFL"): (0.06, 10.0),
    ("Control", "SNCA"): (0.06, 10.0),
    ("ALS", "MAPT"): (0.06, 10.0),
    ("AD", "GFAP"): (0.06, 10.0),
}

# Readings below the QC cutoff are inflated for one group, plus occasional
# spikes. Dropping the QC filter then moves the median, not just the mean.
LOW_QC = 0.7
LOW_QC_BIAS = {"ALS": 3.0}
LOW_QC_PROTEIN_BIAS = {"TREM2": 2.5}
SPIKE_P = 0.03
SPIKE = 8.0


def generate_biomarker_dataset(n=50_000, seed=42, version=2):
    """
    A reproducible synthetic proteomics panel, built so that each way of going
    wrong -- wrong statistic, a dropped filter, a moved cutoff -- changes which
    group wins somewhere. `version=1` is the original, single-signal table.
    """
    if version == 1:
        return generate_biomarker_dataset_v1(n=n, seed=seed)

    rng = np.random.default_rng(seed)

    disease = rng.choice(DISEASES, size=n, p=DISEASE_P)
    protein = rng.choice(PROTEINS, size=n)

    mu = np.array([BASELINE[p][0] for p in protein])
    sigma = np.array([BASELINE[p][1] for p in protein])
    intensity = rng.lognormal(mean=mu, sigma=sigma)

    intensity[disease == "Control"] *= CONTROL_LIFT

    for (d, p), factor in EFFECT.items():
        intensity[(disease == d) & (protein == p)] *= factor

    for (d, p), (rate, factor) in OUTLIERS.items():
        cell = np.flatnonzero((disease == d) & (protein == p))
        hit = rng.choice(cell, size=max(1, int(len(cell) * rate)), replace=False)
        intensity[hit] *= factor

    # one decimal place fewer than the raw float, as an instrument would report
    qc = np.round(rng.uniform(0.5, 1.0, size=n), 2)

    low = qc < LOW_QC
    bias = np.array([LOW_QC_BIAS.get(d, 1.0) for d in disease])
    p_bias = np.array([LOW_QC_PROTEIN_BIAS.get(p, 1.0) for p in protein])
    intensity[low] *= bias[low] * p_bias[low]
    spiked = low & (rng.random(n) < SPIKE_P)
    intensity[spiked] *= SPIKE

    df = pd.DataFrame(
        {
            "sample_id": [f"S{i:05d}" for i in range(n)],
            "disease": disease,
            "protein": protein,
            "intensity": intensity,
            "qc_score": qc,
        }
    )

    missing = rng.choice(df.index, size=max(1, int(n * 0.01)), replace=False)
    df.loc[missing, "intensity"] = np.nan

    return df
