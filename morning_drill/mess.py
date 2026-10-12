"""
Raw data for the cleaning topics.

The clean table is the truth. The student is handed a copy with known defects
applied, and has to clean it back. Every defect here is exactly reversible, so
the expected answer is simply the spec evaluated on the clean table: there is
no judgement about which cleaning choice was "right", and no way for a
reasonable cleaning to land on a different answer.

    sentinel       missing intensities stored as -999 instead of NaN
    duplicates     some rows appear twice, exactly
    protein_text   protein names vary in case and carry stray spaces
    disease_text   disease labels carry stray spaces

The text defects keep the canonical spelling recoverable by strip() and
upper() (protein) or strip() alone (disease), so the label a correct pipeline
returns is the same label the grader expects.
"""

import numpy as np
import pandas as pd

from .graders import compare_answers
from .spec import evaluate_spec

SENTINEL = -999.0

KINDS = {
    "sentinel": "missing intensities are stored as -999 instead of being blank",
    "duplicates": "some rows appear twice",
    "protein_text": "protein names vary in case and carry stray spaces",
    "disease_text": "disease labels carry stray spaces",
}

# What the spec has to do for a defect to matter, in the words the trainer
# agent needs when its challenge is rejected.
ADVICE = {
    "sentinel": "use a statistic the -999 values move (mean, median, std, sum or min)",
    "duplicates": "use a statistic repeated rows move (count, sum, mean, median or std; max and min ignore duplicates)",
    "protein_text": "filter on a protein or group by protein, so the name variants matter",
    "disease_text": "filter on a disease or group by disease, so the label variants matter",
}

# Applied in this order: text first, duplicates last, so a duplicate is an
# exact copy of the row it duplicates.
ORDER = ("protein_text", "disease_text", "sentinel", "duplicates")

# Which topics hand the student the raw table. The rule is the topic, nothing
# else: reaching a cleaning topic is what makes the data messy.
TOPIC_MESS = {
    "missing_duplicates": ("sentinel", "duplicates"),
    "string_ops": ("protein_text", "disease_text"),
}

TEXT_RATE = 0.12
DUPLICATE_RATE = 0.03


def mess_for_topic(topic):
    return TOPIC_MESS.get(topic, ())


def _vary_protein(value, choice):
    return (
        value.lower(),
        value.title(),
        f" {value}",
        f"{value} ",
    )[choice]


def _vary_disease(value, choice):
    return (f" {value}", f"{value} ", f" {value} ")[choice]


def messify(df, kinds, seed=0):
    """
    A copy of `df` with the named defects applied. Deterministic in `seed`, and
    safe on any frame with the biomarker columns, including the planted rows
    the method check adds.
    """
    kinds = [k for k in ORDER if k in set(kinds)]

    if not kinds:
        return df

    rng = np.random.default_rng(seed)
    out = df.copy()

    for kind in kinds:
        if kind in ("protein_text", "disease_text"):
            column = "protein" if kind == "protein_text" else "disease"
            vary = _vary_protein if kind == "protein_text" else _vary_disease
            width = 4 if kind == "protein_text" else 3

            hit = rng.random(len(out)) < TEXT_RATE
            choices = rng.integers(0, width, size=len(out))
            values = out[column].to_numpy(dtype=object)

            for i in np.flatnonzero(hit):
                values[i] = vary(values[i], int(choices[i]))

            out[column] = values

        elif kind == "sentinel":
            out["intensity"] = out["intensity"].fillna(SENTINEL)

        elif kind == "duplicates":
            take = rng.random(len(out)) < DUPLICATE_RATE
            out = pd.concat([out, out[take]], ignore_index=True)

    return out.reset_index(drop=True) if "duplicates" in kinds else out


def cleaning_code(kinds):
    """
    The cleaning steps for the reference solution, as source.
    """
    lines = []

    if "protein_text" in kinds:
        lines.append('df["protein"] = df["protein"].str.strip().str.upper()')

    if "disease_text" in kinds:
        lines.append('df["disease"] = df["disease"].str.strip()')

    if "sentinel" in kinds:
        lines.append(f'df["intensity"] = df["intensity"].replace({SENTINEL}, np.nan)')

    if "duplicates" in kinds:
        lines.append("df = df.drop_duplicates()")

    return "\n".join(lines)


def describe(kinds):
    """The defects, for the trainer agent's prompt."""
    return "\n".join(f"  - {KINDS[k]}" for k in ORDER if k in kinds)


def mess_report(truth, spec, kinds, seed=0):
    """
    Which defects change the answer if left in. A defect that does not is a
    cleaning step the student can skip, and the question does not test it.
    """
    expected = evaluate_spec(truth, spec)
    changes, ignorable = [], []

    for kind in kinds:
        try:
            got = evaluate_spec(messify(truth, [kind], seed), spec)
        except Exception:
            changes.append(kind)
            continue

        (ignorable if compare_answers(got, expected) else changes).append(kind)

    return {"matters": changes, "ignorable": ignorable}
