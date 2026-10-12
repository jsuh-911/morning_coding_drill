"""
The three agents, driven by stub models so none of this needs an API key.
"""

import pandas as pd
import pytest

from morning_drill.agents import author, decide, review, verify
from morning_drill.agents.library import LIBRARY, from_library
from morning_drill.agents.schemas import BlindSolution, Brief, GeneratedChallenge, Review
from morning_drill.agents.trainer_agent import Rejection, validate
from morning_drill.generators import generate_biomarker_dataset
from morning_drill.spec import ChallengeSpec, evaluate_spec, spec_to_reference_code


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
        mission="Among QC-passing GFAP measurements, report the difference between the largest and smallest group-level median intensity across disease groups.",
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
    Every protein has identical QC readings, so its median by group is the
    same whichever statistic a confused student reaches for.
    """
    spec = ChallengeSpec(
        group_by=["protein"], value="qc_score", agg="median", reduce="idxmin"
    )
    bland = challenge_with(
        spec=spec,
        mission="Determine which assay has the weakest median QC score.",
    )

    # every protein has the same QC readings, so no statistic can separate them
    flat = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(10)],
            "disease": ["AD"] * 10,
            "protein": ["MAPT", "SNCA", "NEFL", "GFAP", "TREM2"] * 2,
            "intensity": [1.0] * 10,
            "qc_score": [0.6, 0.7, 0.8, 0.9, 1.0] * 2,
        }
    )

    with pytest.raises(Rejection) as exc:
        validate(bland, flat)

    assert "not discriminating" in exc.value.reason


def test_trainer_agent_revises_after_a_rejection(df):
    bad = challenge_with(
        context="Confirm that AD has the highest median NEFL intensity.",
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

    assert challenge.spec == good.spec
    assert len(notes) == 1
    assert "contains the answer" in notes[0]
    assert "Your challenge was rejected" in model.prompts[1]


def test_trainer_agent_falls_back_to_the_library_when_it_keeps_failing(df):
    bad = challenge_with(context="Confirm that AD has the highest value.", spec=ChallengeSpec(
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

    reference = "answer = " + spec_to_reference_code(challenge.spec).strip()

    good = verify(challenge, df, expected, reference)
    assert good["correct"] is True

    bad = verify(challenge, df, "PD", reference)
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


# ------------------------------------------------- mission matches the spec








def test_every_library_mission_matches_its_spec(df):
    from morning_drill.mess import mess_for_topic

    for challenge in LIBRARY:
        validate(challenge, df, mess_kinds=mess_for_topic(challenge.topic))


# ------------------------------------------------------------ blind solver


class RoutedModel:
    """Answers each structured-output schema from its own queue."""

    def __init__(self, **queues):
        self.queues = {k: list(v) for k, v in queues.items()}
        self.calls = {k: 0 for k in queues}

    def with_structured_output(self, schema):
        outer = self
        name = schema.__name__

        class _S:
            def invoke(self, prompt):
                outer.calls[name] = outer.calls.get(name, 0) + 1
                queue = outer.queues.get(name, [])

                if not queue:
                    raise RuntimeError("queue exhausted")

                return queue.pop(0)

        return _S()


def reading(spec=None, **kw):
    from morning_drill.agents.schemas import BlindSolution

    return BlindSolution(spec=spec, **kw)


def test_blind_reader_who_agrees_lets_the_mission_through(df):
    challenge = challenge_with()
    model = RoutedModel(BlindSolution=[reading(challenge.spec)] * 2)

    report = validate(challenge, df, solver=model)

    assert report["blind_check"] == "agreed"
    assert model.calls["BlindSolution"] == 2


def test_blind_reader_who_cannot_express_the_mission_rejects_it(df):
    """
    "Which protein shows the greatest spread between its highest and lowest
    intensity" while the spec returns one number: a reader cannot write a spec
    for it, and nobody needed a rule about the word "spread" to find out.
    """
    challenge = challenge_with(
        mission=(
            "Restrict to measurements with qc_score above 0.7, then find which "
            "protein shows the greatest spread between its highest and lowest "
            "recorded intensity."
        ),
        spec=ChallengeSpec(
            filters=[{"column": "qc_score", "op": ">", "value": 0.7}],
            group_by=["protein"],
            value="intensity",
            agg="max",
            reduce="diff_extremes",
        ),
    )
    model = RoutedModel(
        BlindSolution=[
            reading(expressible=False, problem="a range within each protein is not an aggregate")
        ]
    )

    with pytest.raises(Rejection) as exc:
        validate(challenge, df, solver=model)

    assert "cannot be answered as worded" in exc.value.reason
    assert "range within each protein" in exc.value.reason


def test_blind_reader_who_returns_a_different_kind_of_answer_rejects_it(df):
    challenge = challenge_with()  # diff_extremes: one number
    label = challenge.spec.model_copy(update={"reduce": "idxmax"})
    model = RoutedModel(BlindSolution=[reading(label)])

    with pytest.raises(Rejection) as exc:
        validate(challenge, df, solver=model)

    assert "group label" in exc.value.reason
    assert "single number" in exc.value.reason


def test_blind_reader_who_computes_a_different_answer_names_the_difference(df):
    challenge = challenge_with()
    other = challenge.spec.model_copy(update={"agg": "mean"})
    model = RoutedModel(BlindSolution=[reading(other)])

    with pytest.raises(Rejection) as exc:
        validate(challenge, df, solver=model)

    assert "differ in agg" in exc.value.reason


def test_blind_reader_who_calls_it_ambiguous_rejects_it(df):
    challenge = challenge_with()
    model = RoutedModel(
        BlindSolution=[reading(challenge.spec, ambiguous=True, problem="median or mean?")]
    )

    with pytest.raises(Rejection) as exc:
        validate(challenge, df, solver=model)

    assert "ambiguous" in exc.value.reason


def test_a_broken_solver_does_not_block_a_good_challenge(df):
    model = RoutedModel(BlindSolution=[])

    report = validate(challenge_with(), df, solver=model)

    assert report["blind_check"] == "skipped"


def test_trainer_agent_revises_after_the_blind_reader_rejects(df):
    bad = challenge_with(mission="Which protein has the largest spread?")
    good = challenge_with()
    model = RoutedModel(
        GeneratedChallenge=[bad, good],
        BlindSolution=[
            reading(expressible=False, problem="no such aggregate"),
            reading(good.spec),
            reading(good.spec),
        ],
    )

    challenge, report, notes = author(
        model,
        Brief(topic="groupby", level=3, rationale="test"),
        df,
        blind_check=True,
    )

    assert challenge.spec == good.spec
    assert any("cannot be answered as worded" in n for n in notes)
    assert report["blind_check"] == "agreed"


def test_the_library_is_not_sent_to_the_solver(df):
    from morning_drill.agents.schemas import BlindSolution

    model = RoutedModel(BlindSolution=[])

    challenge, _, _ = author(None, Brief(topic="groupby", level=1, rationale="t"), df)

    assert challenge in LIBRARY
    assert model.calls["BlindSolution"] == 0


def test_every_library_mission_states_the_numbers_its_spec_filters_on(df):
    """
    A student cannot guess that "QC-passing" means qc_score >= 0.85. Anything
    the grader filters on numerically has to be readable in the mission.
    """
    for challenge in LIBRARY:
        for f in challenge.spec.filters:
            if isinstance(f.value, (int, float)):
                assert str(f.value) in challenge.mission, (
                    f"{challenge.id}: mission never states {f.column} {f.op} {f.value}"
                )


def test_a_reader_who_takes_the_answer_to_be_a_name_rejects_a_numeric_spec(df):
    """
    The case that got through the first version of the solver: shown the
    reducers, a reader maps "spread" onto diff_extremes and calls it fine.
    The answer form is decided from the words alone, before any method.
    """
    from morning_drill.agents.schemas import AnswerForm

    challenge = challenge_with()  # diff_extremes: one number
    model = RoutedModel(
        AnswerForm=[AnswerForm(asks_for="group label")],
        BlindSolution=[reading(challenge.spec)],
    )

    with pytest.raises(Rejection) as exc:
        validate(challenge, df, solver=model)

    assert "takes the answer to be a group label" in exc.value.reason
    assert "single number" in exc.value.reason
    assert model.calls.get("BlindSolution", 0) == 0  # stopped before step 2


def test_a_reader_who_finds_the_answer_unclear_rejects_it(df):
    from morning_drill.agents.schemas import AnswerForm

    model = RoutedModel(
        AnswerForm=[
            AnswerForm(asks_for="single number", clear=False, problem="asks for a name and a value")
        ]
    )

    with pytest.raises(Rejection) as exc:
        validate(challenge_with(), df, solver=model)

    assert "unclear what the answer is" in exc.value.reason


def test_a_reader_who_agrees_on_the_form_goes_on_to_the_computation(df):
    from morning_drill.agents.schemas import AnswerForm

    challenge = challenge_with()
    model = RoutedModel(
        AnswerForm=[AnswerForm(asks_for="single number")] * 2,
        BlindSolution=[reading(challenge.spec)] * 2,
    )

    assert validate(challenge, df, solver=model)["blind_check"] == "agreed"


# ------------------------------------------------- malformed model output


def test_a_malformed_challenge_is_retried_not_treated_as_an_outage(df):
    """
    A spec the schema refuses (diff_extremes on a count) or a missing field
    raises ValueError at parse time. The model is told what was wrong and asked
    again; it does not silently fall back to the library.
    """
    good = challenge_with()

    class Flaky:
        def __init__(self):
            self.prompts = []
            self.queue = [ValueError("1 validation error for GeneratedChallenge\ntitle\n  Field required"), good]

        def with_structured_output(self, schema):
            outer = self

            class _S:
                def invoke(self, prompt):
                    outer.prompts.append(prompt)
                    item = outer.queue.pop(0)
                    if isinstance(item, Exception):
                        raise item
                    return item

            return _S()

    model = Flaky()

    challenge, _, notes = author(model, Brief(topic="groupby", level=3, rationale="t"), df)

    assert challenge.spec == good.spec
    assert any("not a valid challenge" in n for n in notes)
    assert "Field required" in model.prompts[1]
    assert "could not be read as a challenge" in model.prompts[1]


def test_an_outage_still_falls_back_to_the_library(df):
    class Down:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            raise RuntimeError("api down")

    challenge, _, notes = author(Down(), Brief(topic="groupby", level=1, rationale="t"), df)

    assert challenge in LIBRARY
    assert any("unavailable" in n for n in notes)


def test_a_rejected_cleaning_challenge_says_what_each_defect_needs(df):
    from morning_drill.mess import mess_for_topic

    spec = ChallengeSpec(
        filters=[{"column": "qc_score", "op": ">=", "value": 0.8}],
        group_by=["protein"],
        value="intensity",
        agg="median",
        reduce="nlargest",
    )
    challenge = challenge_with(
        mission="Clean the raw export, then return the three proteins with the highest median intensity at qc_score >= 0.8.",
        spec=spec,
        topic="string_ops",
        level=3,
    )

    with pytest.raises(Rejection) as exc:
        validate(challenge, df, mess_kinds=mess_for_topic("string_ops"))

    assert "disease_text: filter on a disease or group by disease" in exc.value.reason


def test_the_challenge_takes_its_topic_and_level_from_the_brief(df):
    """
    A model that labels its challenge sql/joins must not get it recorded,
    scored and displayed as sql/joins when the brief said groupby level 3.
    """
    drifted = challenge_with(topic="joins", level=1)
    model = ScriptedModel([drifted])

    challenge, _, _ = author(
        model, Brief(topic="groupby", level=3, rationale="x"), df
    )

    assert (challenge.topic, challenge.level) == ("groupby", 3)
