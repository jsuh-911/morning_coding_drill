import numpy as np
import pandas as pd


def generate_biomarker_dataset(n=50_000, seed=42):
    """
    Generate a reproducible synthetic biomarker dataset.
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
