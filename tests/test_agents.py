"""
The three agents, driven by stub models so none of this needs an API key.
"""

import pandas as pd
import pytest

from morning_drill.agents import author, decide, review, verify
from morning_drill.agents.library import LIBRARY, from_library
from morning_drill.agents.schemas import Brief, GeneratedChallenge, Review
from morning_drill.agents.trainer_agent import Rejection, validate
from morning_drill.generators import generate_biomarker_dataset
from morning_drill.spec import ChallengeSpec, evaluate_spec


@pytest.fixture(scope="module")
def df():
    return generate_biomarker_dataset(n=20_000, seed=3)


class ScriptedModel:
    """Returns a queued object per structured call, and records prompts."""

    def __init__(self, queue):
        self.queue = list(queue)
        self.prompts = []

    def with_structured_output(self, schema):
        outer = self

        class _S:
            def invoke(self, prompt):
                outer.prompts.append(prompt)

                if not outer.queue:
                    raise RuntimeError("queue exhausted")

                return outer.queue.pop(0)

        return _S()

    def invoke(self, prompt):
        self.prompts.append(prompt)
        raise RuntimeError("not used in these tests")


def challenge_with(**kw):
    base = dict(
        title="T",
        mission="Identify which disease group has the highest median NEFL intensity among QC-passing samples.",
        toolbox=[],
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
        level=3,
    )
    base.update(kw)
    return GeneratedChallenge(**base)


# ------------------------------------------------------------ trainer agent


def test_validate_rejects_a_mission_that_states_the_answer(df):
    spec = ChallengeSpec(
        filters=[
            {"column": "qc_score", "op": ">=", "value": 0.85},
            {"column": "protein", "op": "==", "value": "NEFL"},
        ],
        group_by=["disease"],
        value="intensity",
        agg="median",
        reduce="idxmax",
    )
    bad = challenge_with(
        mission="Confirm that AD has the highest median NEFL intensity.",
        spec=spec,
    )

    with pytest.raises(Rejection) as exc:
        validate(bad, df)

    assert "contains the answer" in exc.value.reason


def test_validate_rejects_an_empty_selection(df):
    spec = ChallengeSpec(
        filters=[
            {"column": "disease", "op": "==", "value": "AD"},
            {"column": "disease", "op": "==", "value": "PD"},
        ],
        group_by=["protein"],
        value="intensity",
        agg="median",
    )

    with pytest.raises(Rejection) as exc:
        validate(challenge_with(spec=spec), df)

    assert "select no rows" in exc.value.reason


def test_validate_rejects_a_toolbox_at_high_level(df):
    with pytest.raises(Rejection) as exc:
        validate(challenge_with(level=3, toolbox=["groupby"]), df)

    assert "toolbox" in exc.value.reason


def test_validate_rejects_an_undiscriminating_challenge(df):
    """
    qc_score is uniform noise, so its median by group is the same whichever
    statistic a confused student reaches for.
    """
    spec = ChallengeSpec(
        group_by=["protein"], value="qc_score", agg="median", reduce="idxmin"
    )
    bland = challenge_with(
        spec=spec,
        mission="Determine which assay has the weakest median QC score.",
    )

    with pytest.raises(Rejection) as exc:
        validate(bland, df)

    assert "not discriminating" in exc.value.reason


def test_trainer_agent_revises_after_a_rejection(df):
    bad = challenge_with(
        mission="Confirm that AD has the highest median NEFL intensity.",
        spec=ChallengeSpec(
            filters=[
                {"column": "qc_score", "op": ">=", "value": 0.85},
                {"column": "protein", "op": "==", "value": "NEFL"},
            ],
            group_by=["disease"],
            value="intensity",
            agg="median",
            reduce="idxmax",
        ),
    )
    good = challenge_with()

    model = ScriptedModel([bad, good])
    challenge, report, notes = author(
        model, Brief(topic="groupby", level=3, rationale="x"), df
    )

    assert challenge.mission == good.mission
    assert len(notes) == 1
    assert "contains the answer" in notes[0]
    assert "Your challenge was rejected" in model.prompts[1]


def test_trainer_agent_falls_back_to_the_library_when_it_keeps_failing(df):
    bad = challenge_with(mission="Confirm that AD has the highest value.", spec=ChallengeSpec(
        filters=[{"column": "protein", "op": "==", "value": "NEFL"},
                 {"column": "qc_score", "op": ">=", "value": 0.85}],
        group_by=["disease"], value="intensity", agg="median", reduce="idxmax"))

    model = ScriptedModel([bad, bad, bad])
    challenge, report, notes = author(
        model, Brief(topic="groupby", level=1, rationale="x"), df
    )

    assert challenge in LIBRARY
    assert any("could not produce" in n for n in notes)


def test_trainer_agent_with_no_model_uses_the_library(df):
    challenge, report, notes = author(
        None, Brief(topic="groupby", level=2, rationale="x"), df
    )

    assert challenge in LIBRARY
    assert challenge.level == 2
    assert notes == []


def test_library_respects_exclusions():
    first = from_library("groupby", 1)
    second = from_library("groupby", 1, exclude=(first.id,))
    assert second.id != first.id


# ------------------------------------------------------------- grader agent


def test_verify_is_the_authority_on_correctness(df):
    challenge = from_library("groupby", 1)
    expected = evaluate_spec(df, challenge.spec)

    good = verify(challenge, df, expected, 'answer = df.groupby("disease")["intensity"].median().idxmax()')
    assert good["correct"] is True

    bad = verify(challenge, df, "PD", 'answer = df.groupby("disease")["intensity"].median().idxmax()')
    assert bad["correct"] is False


def test_verify_catches_a_typed_in_answer(df):
    challenge = from_library("groupby", 1)
    expected = evaluate_spec(df, challenge.spec)

    result = verify(challenge, df, expected, f'answer = {expected!r}')

    assert result["correct"] is False
    assert result["hardcoded"] is True
    assert result["discrepancy"] == {}


def test_verify_reports_constraint_violations(df):
    challenge = from_library("groupby", 1)
    expected = evaluate_spec(df, challenge.spec)

    result = verify(
        challenge,
        df,
        expected,
        "answer = None\nfor d in df.disease.unique():\n    answer = d",
    )

    assert any("for-loop" in v for v in result["violations"])


def test_grader_agent_cannot_call_a_wrong_answer_correct(df):
    challenge = from_library("groupby", 1)
    verification = verify(challenge, df, "PD", "answer = x")

    model = ScriptedModel(
        [
            Review(
                error_class="correct",
                root_cause="looks right to me",
                reasoning_sound=True,
            )
        ]
    )

    result = review(model, challenge, "answer = x", "'PD'", verification, 1)

    assert result.error_class != "correct"


def test_grader_agent_downgrades_a_correct_answer_that_broke_a_constraint(df):
    challenge = from_library("groupby", 3)   # this one forbids .apply()
    assert "no_apply" in challenge.constraint_ids
    expected = evaluate_spec(df, challenge.spec)

    verification = verify(
        challenge,
        df,
        expected,
        'answer = df.groupby("disease")["intensity"].apply(lambda s: s.median()).idxmax()',
    )
    verification["correct"] = True

    model = ScriptedModel(
        [Review(error_class="correct", root_cause="", reasoning_sound=True)]
    )

    result = review(model, challenge, "x", "x", verification, 1)

    assert result.error_class == "style_only"


def test_grader_agent_hint_strength_follows_the_attempt(df):
    challenge = from_library("groupby", 1)
    verification = verify(challenge, df, "PD", "answer = x")

    for attempt, marker in [
        (1, "No code, not even a fragment"),
        (2, "missed it twice"),
        (3, "Stop hinting"),
    ]:
        model = ScriptedModel(
            [Review(error_class="conceptual", root_cause="x", reasoning_sound=False)]
        )
        review(model, challenge, "answer = x", "'PD'", verification, attempt)
        assert marker in model.prompts[0]


def test_grader_agent_falls_back_when_the_model_breaks(df):
    class Broken:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            raise RuntimeError("api down")

    challenge = from_library("groupby", 1)
    verification = verify(challenge, df, "PD", "answer = x")

    result = review(Broken(), challenge, "answer = x", "'PD'", verification, 1)

    assert isinstance(result, Review)
    assert result.error_class != "correct"


# ----------------------------------------------------------- progress agent


BASE_PROGRESS = {
    "python": {
        "dataframe_filtering": {"status": "mastered", "level": 4},
        "groupby": {"status": "active", "level": 1, "attempts": 0, "correct": 0},
    }
}


def test_clean_pass_raises_the_level():
    decision = decide(
        None,
        BASE_PROGRESS,
        outcome={"topic": "groupby", "level": 1, "passed": True, "attempt": 1},
    )

    assert decision.next_level == 2
    assert decision.topic_status == "active"


def test_three_attempts_means_needs_rework_and_no_level_rise():
    decision = decide(
        None,
        BASE_PROGRESS,
        outcome={"topic": "groupby", "level": 2, "passed": True, "attempt": 3},
    )

    assert decision.topic_status == "needs_rework"
    assert decision.next_level == 2


def test_mastery_requires_surviving_level_four():
    at_three = decide(
        None,
        BASE_PROGRESS,
        outcome={"topic": "groupby", "level": 3, "passed": True, "attempt": 1},
    )
    assert at_three.topic_status != "mastered"

    at_four = decide(
        None,
        BASE_PROGRESS,
        outcome={"topic": "groupby", "level": 4, "passed": True, "attempt": 1},
    )
    assert at_four.topic_status == "mastered"


def test_failure_holds_the_level():
    decision = decide(
        None,
        BASE_PROGRESS,
        outcome={"topic": "groupby", "level": 2, "passed": False, "attempt": 1},
    )

    assert decision.next_level == 2


def test_delayed_retrieval_is_scheduled_periodically():
    history = [
        {"topic": "groupby", "level": i, "passed": True, "attempt": 1}
        for i in range(1, 5)
    ]

    decision = decide(
        None,
        BASE_PROGRESS,
        outcome={"topic": "groupby", "level": 2, "passed": True, "attempt": 1},
        history=history,
    )

    assert decision.brief.level == 4
    assert decision.brief.topic == "dataframe_filtering"
    assert "groupby" in decision.brief.fold_in


def test_an_invented_topic_is_corrected():
    from morning_drill.agents.schemas import Brief as B
    from morning_drill.agents.schemas import ProgressDecision

    model = ScriptedModel(
        [
            ProgressDecision(
                topic_status="active",
                next_level=2,
                note="",
                brief=B(topic="quantum_pandas", level=2, rationale="made up"),
            )
        ]
    )

    decision = decide(model, BASE_PROGRESS)

    assert decision.brief.topic == "groupby"


def test_validate_rejects_a_mission_asking_for_two_outputs(df):
    """
    Observed with a real model: the mission asked to "report both the winning
    disease and its median intensity value for each protein", but the spec
    returns a single ('NEFL', 'AD') tuple.
    """
    spec = ChallengeSpec(
        filters=[{"column": "qc_score", "op": ">=", "value": 0.8}],
        group_by=["protein", "disease"],
        value="intensity",
        agg="median",
        reduce="idxmax",
    )
    bad = challenge_with(
        spec=spec,
        mission=(
            "Within each protein, identify the disease with the highest median "
            "intensity. Report both the winning disease and its median value."
        ),
    )

    with pytest.raises(Rejection) as exc:
        validate(bad, df)

    assert "asks for two things" in exc.value.reason


def test_a_series_returning_spec_may_ask_for_values(df):
    spec = ChallengeSpec(
        filters=[{"column": "qc_score", "op": ">=", "value": 0.85},
                 {"column": "disease", "op": "==", "value": "AD"}],
        group_by=["protein"],
        value="intensity",
        agg="std",
        reduce="nlargest",
        n=3,
    )
    ok = challenge_with(
        spec=spec,
        mission="Return the three most variable proteins and their values.",
    )

    validate(ok, df)   # must not raise
