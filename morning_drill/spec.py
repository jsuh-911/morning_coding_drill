"""
Machine-checkable challenge specs.

The trainer agent invents the question. It does NOT invent the grading logic:
it emits a declarative spec, and the code in this module derives the expected
answer from that spec with pandas. So a generated challenge is as verifiable
as a hand-written one, and no model-authored code ever runs.

A spec is validated against the dataset's declared schema, so a malformed or
adversarial spec is rejected rather than executed.
"""

from typing import List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


# ----------------------------------------------------------------- datasets

DATASETS = {
    "biomarker": {
        "generator": "biomarker",
        "columns": {
            "sample_id": "id",
            "disease": "categorical",
            "protein": "categorical",
            "intensity": "numeric",
            "qc_score": "numeric",
        },
        "categories": {
            "disease": ["Control", "AD", "PD", "ALS"],
            "protein": ["MAPT", "SNCA", "NEFL", "GFAP", "TREM2"],
        },
    },
}

FILTER_OPS = {
    ">=": lambda s, v: s >= v,
    ">": lambda s, v: s > v,
    "<=": lambda s, v: s <= v,
    "<": lambda s, v: s < v,
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    "in": lambda s, v: s.isin(v),
    "not in": lambda s, v: ~s.isin(v),
    "notna": lambda s, v: s.notna(),
    "isna": lambda s, v: s.isna(),
}

AGGS = [
    "median",
    "mean",
    "max",
    "min",
    "sum",
    "count",
    "std",
    "nunique",
]

REDUCERS = [
    "idxmax",
    "idxmin",
    "max",
    "min",
    "none",
    "nlargest",
    "nsmallest",
    "diff_extremes",
]


# -------------------------------------------------------------------- model


class Filter(BaseModel):
    column: str
    op: Literal[tuple(FILTER_OPS)]  # type: ignore[valid-type]
    value: Optional[object] = None


class ChallengeSpec(BaseModel):
    """
    A pandas pipeline, declared rather than coded.

        filters -> groupby(group_by)[value].agg(agg) -> reduce
    """

    dataset: Literal["biomarker"] = "biomarker"
    filters: List[Filter] = Field(default_factory=list, max_length=4)
    group_by: List[str] = Field(min_length=1, max_length=2)
    value: str
    agg: Literal[tuple(AGGS)]  # type: ignore[valid-type]
    reduce: Literal[tuple(REDUCERS)] = "idxmax"  # type: ignore[valid-type]
    n: int = Field(default=3, ge=1, le=10)

    @field_validator("group_by")
    @classmethod
    def _group_by_is_categorical(cls, v):
        columns = DATASETS["biomarker"]["columns"]

        for col in v:
            if columns.get(col) != "categorical":
                raise ValueError(
                    f"group_by must be categorical; {col!r} is "
                    f"{columns.get(col, 'unknown')}"
                )

        return v

    @field_validator("value")
    @classmethod
    def _value_is_numeric(cls, v):
        columns = DATASETS["biomarker"]["columns"]

        if columns.get(v) != "numeric":
            raise ValueError(
                f"value must be numeric; {v!r} is {columns.get(v, 'unknown')}"
            )

        return v

    @model_validator(mode="after")
    def _filters_reference_real_columns(self):
        columns = DATASETS[self.dataset]["columns"]
        categories = DATASETS[self.dataset].get("categories", {})

        for f in self.filters:
            if f.column not in columns:
                raise ValueError(f"unknown filter column: {f.column!r}")

            if f.op in ("notna", "isna"):
                continue

            if f.value is None:
                raise ValueError(f"filter on {f.column!r} needs a value")

            allowed = categories.get(f.column)

            if allowed is not None:
                values = f.value if isinstance(f.value, list) else [f.value]

                for value in values:
                    if value not in allowed:
                        raise ValueError(
                            f"{value!r} is not a value of {f.column!r}; "
                            f"allowed: {allowed}"
                        )

        if self.reduce == "diff_extremes" and self.agg == "count":
            raise ValueError("diff_extremes is not meaningful on a count")

        return self

    # ------------------------------------------------------------ behaviour

    @property
    def signature(self):
        """
        Stable identity of the computation, for telling the trainer agent
        what it has already set. Independent of how the mission is worded.
        """
        filters = sorted(
            f"{f.column}{f.op}{f.value}" if f.value is not None else f"{f.column}.{f.op}"
            for f in self.filters
        )

        return "|".join(
            [
                "+".join(self.group_by),
                self.value,
                self.agg,
                self.reduce if self.reduce != "none" else "none",
                ",".join(filters) or "nofilter",
            ]
        )

    @property
    def answer_kind(self):
        """
        What shape of object a correct answer has. Drives the constraint
        text shown to the student and the structural report on a miss.
        """
        if self.reduce in ("idxmax", "idxmin"):
            return "group label" if len(self.group_by) == 1 else "group key tuple"

        if self.reduce in ("max", "min", "diff_extremes"):
            return "single number"

        return "Series indexed by group"


def evaluate_spec(df, spec):
    """
    Compute the expected answer. This is the ground truth.
    """
    frame = df

    for f in spec.filters:
        frame = frame[FILTER_OPS[f.op](frame[f.column], f.value)]

    key = spec.group_by[0] if len(spec.group_by) == 1 else spec.group_by
    grouped = frame.groupby(key)[spec.value].agg(spec.agg)

    if spec.reduce == "none":
        return grouped
    if spec.reduce == "idxmax":
        return grouped.idxmax()
    if spec.reduce == "idxmin":
        return grouped.idxmin()
    if spec.reduce == "max":
        return grouped.max()
    if spec.reduce == "min":
        return grouped.min()
    if spec.reduce == "nlargest":
        return grouped.nlargest(spec.n)
    if spec.reduce == "nsmallest":
        return grouped.nsmallest(spec.n)
    if spec.reduce == "diff_extremes":
        return grouped.max() - grouped.min()

    raise ValueError(f"unknown reduce: {spec.reduce}")


def _filter_code(f):
    if f.op == "notna":
        return f'df["{f.column}"].notna()'
    if f.op == "isna":
        return f'df["{f.column}"].isna()'
    if f.op == "in":
        return f'df["{f.column}"].isin({f.value!r})'
    if f.op == "not in":
        return f'~df["{f.column}"].isin({f.value!r})'

    return f'(df["{f.column}"] {f.op} {f.value!r})'


def spec_to_reference_code(spec):
    """
    The reference solution, as source. Generated from the spec, so it cannot
    drift from what the grader computes.
    """
    if spec.filters:
        mask = " & ".join(_filter_code(f) for f in spec.filters)
        head = f"    df.loc[{mask}]\n"
    else:
        head = "    df\n"

    key = (
        f'"{spec.group_by[0]}"'
        if len(spec.group_by) == 1
        else repr(spec.group_by)
    )

    body = f'      .groupby({key})["{spec.value}"]\n      .{spec.agg}()\n'

    tail = {
        "none": "",
        "idxmax": "      .idxmax()\n",
        "idxmin": "      .idxmin()\n",
        "max": "      .max()\n",
        "min": "      .min()\n",
        "nlargest": f"      .nlargest({spec.n})\n",
        "nsmallest": f"      .nsmallest({spec.n})\n",
    }.get(spec.reduce, "")

    code = f"answer = (\n{head}{body}{tail})"

    if spec.reduce == "diff_extremes":
        code = (
            f"grouped = (\n{head}{body})\n\n"
            "answer = grouped.max() - grouped.min()"
        )

    return code
