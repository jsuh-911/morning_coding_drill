"""
compare_answers must return a bool for every submission shape.

Regression: a learner who forgot .idxmax() submitted a Series, the final
`answer == expected` produced an elementwise Series, and the `if success:`
that consumed it raised "The truth value of a Series is ambiguous".
"""

import numpy as np
import pandas as pd
import pytest

from morning_drill.generators import generate_biomarker_dataset
from morning_drill.graders import compare_answers, discrepancy_report
from morning_drill.spec import ChallengeSpec, evaluate_spec


SPEC = ChallengeSpec(
    filters=[
        {"column": "qc_score", "op": ">=", "value": 0.85},
        {"column": "protein", "op": "==", "value": "NEFL"},
    ],
    group_by=["disease"],
    value="intensity",
    agg="median",
    reduce="idxmax",
)


@pytest.fixture(scope="module")
def df():
    return generate_biomarker_dataset(n=5_000, seed=1)


def test_the_right_scalar_compares_equal(df):
    expected = evaluate_spec(df, SPEC)
    assert compare_answers(expected, expected) is True


@pytest.mark.parametrize(
    "answer",
    [
        pd.Series([1.0, 2.0], index=["AD", "PD"]),
        pd.DataFrame({"a": [1, 2]}),
        np.array(["AD", "PD"]),
        pd.Index(["AD", "PD"]),
        ["AD", "PD"],
        ("AD", "PD"),
        {"AD"},
        {"AD": 1},
        None,
        0.5,
    ],
)
def test_wrong_shapes_return_false_not_raise(answer, df):
    result = compare_answers(answer, evaluate_spec(df, SPEC))
    assert isinstance(result, bool)
    assert result is False


def test_series_expected_against_scalar_answer():
    assert compare_answers(3.0, pd.Series([1.0, 2.0])) is False


def test_frame_expected_against_series_answer():
    assert compare_answers(pd.Series([1]), pd.DataFrame({"a": [1]})) is False


def test_matching_series_compares_equal():
    expected = pd.Series([1.0, 2.0], index=["AD", "PD"])
    assert compare_answers(expected.copy(), expected) is True


def test_discrepancy_names_the_missing_reduction(df):
    expected = evaluate_spec(df, SPEC)
    submitted = evaluate_spec(df, SPEC.model_copy(update={"reduce": "none"}))

    notes = " ".join(discrepancy_report(submitted, expected)["notes"])

    assert "Series" in notes
    assert "reduction" in notes or "aggregation" in notes


def test_discrepancy_leaks_no_values(df):
    expected = evaluate_spec(df, SPEC)
    report = discrepancy_report(pd.Series([1.0]), expected)

    assert expected not in str(report)


# ------------------------------------------------------------ group key tuple


def test_a_two_column_group_key_is_compared_elementwise():
    from morning_drill.graders import compare_answers

    assert compare_answers(("AD", "NEFL"), ("AD", "NEFL"))
    assert compare_answers(["AD", "NEFL"], ("AD", "NEFL"))
    assert not compare_answers(("AD", "GFAP"), ("AD", "NEFL"))
    assert not compare_answers(("AD",), ("AD", "NEFL"))
    assert not compare_answers("AD", ("AD", "NEFL"))
    assert not compare_answers(("AD", "NEFL", "x"), ("AD", "NEFL"))


def test_a_two_column_idxmax_question_can_actually_be_passed():
    """
    It could not: the right tuple was rejected as "a container where a scalar
    was wanted".
    """
    from morning_drill.agents.grader_agent import verify
    from morning_drill.agents.schemas import GeneratedChallenge
    from morning_drill.spec import ChallengeSpec, evaluate_spec, spec_to_reference_code

    df = generate_biomarker_dataset(n=10_000, seed=2)
    spec = ChallengeSpec(
        filters=[{"column": "qc_score", "op": ">=", "value": 0.7}],
        group_by=["disease", "protein"],
        value="intensity",
        agg="median",
        reduce="idxmax",
    )
    challenge = GeneratedChallenge(title="t", mission="m", spec=spec, topic="groupby", level=2)
    ns = {"df": df}
    exec(spec_to_reference_code(spec), ns)

    assert isinstance(ns["answer"], tuple)
    assert verify(challenge, df, ns["answer"], spec_to_reference_code(spec))["correct"] is True
