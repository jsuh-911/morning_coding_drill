"""
A right value is not a right method.

On the biomarker data AD has the highest median NEFL intensity, and it also
has it with the NEFL filter dropped, with the mean, and with the QC cutoff
moved. These tests pin that: such code must not be graded correct.
"""

import pandas as pd
import pytest

from morning_drill.agents.grader_agent import verify
from morning_drill.agents.library import LIBRARY
from morning_drill.generators import generate_biomarker_dataset
from morning_drill.probes import method_failures
from morning_drill.spec import (
    ChallengeSpec,
    Filter,
    evaluate_spec,
    spec_to_reference_code,
)


@pytest.fixture(scope="module")
def df():
    return generate_biomarker_dataset()


@pytest.fixture(scope="module")
def spec():
    return ChallengeSpec(
        filters=[
            Filter(column="qc_score", op=">=", value=0.85),
            Filter(column="protein", op="==", value="NEFL"),
        ],
        group_by=["disease"],
        value="intensity",
        agg="median",
        reduce="idxmax",
    )


RIGHT = "answer = df[(df.qc_score >= 0.85) & (df.protein == 'NEFL')].groupby('disease').intensity.median().idxmax()"

WRONG = {
    "NEFL filter missing": "answer = df[df.qc_score >= 0.85].groupby('disease').intensity.median().idxmax()",
    "QC filter missing": "answer = df[df.protein == 'NEFL'].groupby('disease').intensity.median().idxmax()",
    "mean for median": "answer = df[(df.qc_score >= 0.85) & (df.protein == 'NEFL')].groupby('disease').intensity.mean().idxmax()",
    "> for >=": "answer = df[(df.qc_score > 0.85) & (df.protein == 'NEFL')].groupby('disease').intensity.median().idxmax()",
    "cutoff 0.8": "answer = df[(df.qc_score >= 0.8) & (df.protein == 'NEFL')].groupby('disease').intensity.median().idxmax()",
}


def test_the_premise_holds_every_wrong_pipeline_returns_the_right_label(df, spec):
    truth = evaluate_spec(df, spec)

    for name, code in WRONG.items():
        ns = {"df": df}
        exec(code, ns)
        assert ns["answer"] == truth, name


@pytest.mark.parametrize("name", WRONG)
def test_wrong_method_is_caught(df, spec, name):
    assert method_failures(WRONG[name], df, spec)


def test_right_method_passes(df, spec):
    assert method_failures(RIGHT, df, spec) == []


def test_a_different_valid_route_passes(df, spec):
    code = (
        "answer = (df.query(\"qc_score >= 0.85 and protein == 'NEFL'\")"
        ".groupby('disease')['intensity'].median().sort_values().index[-1])"
    )
    assert method_failures(code, df, spec) == []


def test_a_notebook_cell_ending_in_submit_still_checks(df, spec):
    assert method_failures(RIGHT + "\nsession.submit(answer)", df, spec) == []
    assert method_failures(WRONG["NEFL filter missing"] + "\nsession.submit(answer)", df, spec)


def test_code_that_cannot_be_run_is_not_failed(df, spec):
    assert method_failures("answer = undefined_name", df, spec) == []


def test_verify_marks_the_right_value_by_the_wrong_route_incorrect(df, spec):
    challenge = next(c for c in LIBRARY if c.level == 1).model_copy(update={"spec": spec})
    expected = evaluate_spec(df, spec)

    result = verify(challenge, df, expected, WRONG["NEFL filter missing"])

    assert result["correct"] is False
    assert any("protein" in v for v in result["violations"])
    assert result["discrepancy"] == {}

    assert verify(challenge, df, expected, RIGHT)["correct"] is True


@pytest.mark.parametrize("challenge", LIBRARY, ids=lambda c: c.id)
def test_every_library_reference_solution_passes_its_own_probes(df, challenge):
    code = "answer = " + spec_to_reference_code(challenge.spec).strip()
    assert method_failures(code, df, challenge.spec) == [], challenge.id
