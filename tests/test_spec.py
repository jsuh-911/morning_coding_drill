"""
The spec is the ground truth, so it gets the strictest tests.
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from morning_drill.agents.library import LIBRARY
from morning_drill.generators import generate_biomarker_dataset
from morning_drill.spec import (
    ChallengeSpec,
    evaluate_spec,
    spec_to_reference_code,
)


@pytest.fixture(scope="module")
def df():
    return generate_biomarker_dataset(n=8_000, seed=7)


def test_rejects_non_categorical_group_by():
    with pytest.raises(ValidationError):
        ChallengeSpec(group_by=["intensity"], value="intensity", agg="mean")


def test_rejects_non_numeric_value():
    with pytest.raises(ValidationError):
        ChallengeSpec(group_by=["disease"], value="protein", agg="mean")


def test_rejects_unknown_column():
    with pytest.raises(ValidationError):
        ChallengeSpec(group_by=["disease"], value="nope", agg="mean")


def test_rejects_value_outside_the_category():
    with pytest.raises(ValidationError):
        ChallengeSpec(
            group_by=["disease"],
            value="intensity",
            agg="mean",
            filters=[{"column": "protein", "op": "==", "value": "TAU"}],
        )


def test_rejects_filter_without_a_value():
    with pytest.raises(ValidationError):
        ChallengeSpec(
            group_by=["disease"],
            value="intensity",
            agg="mean",
            filters=[{"column": "qc_score", "op": ">="}],
        )


@pytest.mark.parametrize("challenge", LIBRARY, ids=lambda c: c.title)
def test_reference_code_computes_what_the_spec_says(challenge, df):
    """
    The reference solution is generated from the spec, so it must produce
    exactly what the spec evaluates to. If these ever diverge, the student is
    shown a solution that would not have passed.
    """
    expected = evaluate_spec(df, challenge.spec)

    namespace = {"df": df, "pd": pd}
    exec(spec_to_reference_code(challenge.spec), namespace)  # noqa: S102
    produced = namespace["answer"]

    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(produced, expected)
    elif isinstance(expected, float):
        assert abs(produced - expected) < 1e-9
    else:
        assert produced == expected


def test_every_library_challenge_is_answerable(df):
    for challenge in LIBRARY:
        answer = evaluate_spec(df, challenge.spec)
        assert answer is not None

        if isinstance(answer, pd.Series):
            assert not answer.empty


def test_answer_kind_tracks_the_reducer():
    base = dict(group_by=["disease"], value="intensity", agg="median")

    assert "label" in ChallengeSpec(**base, reduce="idxmax").answer_kind
    assert "number" in ChallengeSpec(**base, reduce="max").answer_kind
    assert "Series" in ChallengeSpec(**base, reduce="none").answer_kind


# ------------------------------------------------------------ range


def test_range_is_the_gap_within_each_group_not_between_groups(df):
    """
    "For each disease group, the gap between its highest and lowest intensity,
    then the largest of those" -- the question that was marked wrong because
    only a between-groups reduce existed.
    """
    spec = ChallengeSpec(
        filters=[{"column": "qc_score", "op": ">", "value": 0.6}],
        group_by=["disease"],
        value="intensity",
        agg="range",
        reduce="max",
    )

    kept = df[df.qc_score > 0.6]
    by_hand = (
        kept.groupby("disease").intensity.max() - kept.groupby("disease").intensity.min()
    ).max()

    assert evaluate_spec(df, spec) == by_hand

    between = ChallengeSpec(
        filters=spec.filters, group_by=["disease"], value="intensity",
        agg="max", reduce="diff_extremes",
    )
    assert evaluate_spec(df, between) != by_hand


def test_the_reference_code_for_range_computes_what_the_spec_does(df):
    from morning_drill.spec import spec_to_reference_code

    spec = ChallengeSpec(
        filters=[{"column": "qc_score", "op": ">", "value": 0.6}],
        group_by=["disease"],
        value="intensity",
        agg="range",
        reduce="max",
    )
    ns = {"df": df}
    exec(spec_to_reference_code(spec), ns)

    assert ns["answer"] == evaluate_spec(df, spec)
