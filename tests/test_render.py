"""
The question is written by code from the spec, so it cannot disagree with it.
"""

import itertools

import pytest

from morning_drill.agents.schemas import GeneratedChallenge
from morning_drill.agents.trainer_agent import Rejection, validate
from morning_drill.generators import generate_biomarker_dataset
from morning_drill.mess import TOPIC_MESS
from morning_drill.render import render_challenge, render_mission, render_title
from morning_drill.spec import AGGS, REDUCERS, ChallengeSpec, Filter


def spec(**kw):
    base = dict(
        filters=[Filter(column="qc_score", op=">", value=0.6)],
        group_by=["disease"],
        value="intensity",
        agg="range",
        reduce="max",
    )
    base.update(kw)
    return ChallengeSpec(**base)


def test_the_question_that_was_marked_wrong_is_now_stated_exactly():
    text = render_mission(spec())

    assert text == (
        "Use only rows where qc_score is strictly above 0.6. "
        "For each disease group, compute the range (highest value minus lowest "
        "value) of intensity. Report the highest of these values."
    )


def test_the_same_spec_always_renders_the_same_text():
    assert render_mission(spec()) == render_mission(spec())


# --------------------------------------------- every part of the spec is stated


COMBOS = [
    (agg, reduce)
    for agg, reduce in itertools.product(AGGS, REDUCERS)
    if not (reduce == "diff_extremes" and agg == "count")
]


@pytest.mark.parametrize("agg,reduce", COMBOS)
def test_every_statistic_and_reduce_is_stated(agg, reduce):
    s = spec(agg=agg, reduce=reduce)
    text = render_mission(s)

    assert "qc_score is strictly above 0.6" in text
    assert "For each disease group" in text

    if agg in ("count", "nunique"):
        assert "count" in text
    else:
        assert "intensity" in text


@pytest.mark.parametrize("reduce", REDUCERS)
def test_what_to_report_matches_the_kind_of_answer(reduce):
    s = spec(agg="mean", reduce=reduce)
    text = render_mission(s)

    if s.answer_form == "group label":
        assert "Report which disease group" in text
    else:
        assert "Report which" not in text

    if s.answer_form == "ranked groups":
        assert "ordered from" in text and str(s.n) in text

    if s.answer_form == "value per group":
        assert "as a Series" in text


@pytest.mark.parametrize("reduce", REDUCERS)
def test_the_title_never_asks_a_different_question_from_the_mission(reduce):
    s = spec(agg="mean", reduce=reduce)
    title = render_title(s)

    assert 0 < len(title) <= 60
    assert not title.lower().startswith("which")

    if s.answer_form == "group label":
        assert title.endswith("by disease group")
    else:
        assert "by disease group" not in title


def test_the_difference_between_range_and_diff_extremes_is_spelled_out():
    within = render_mission(spec(agg="range", reduce="max"))
    between = render_mission(spec(agg="max", reduce="diff_extremes"))

    assert "range (highest value minus lowest value)" in within
    assert "difference between the highest and the lowest of these values across disease groups" in between
    assert within != between


@pytest.mark.parametrize(
    "op,value,phrase",
    [
        (">=", 0.85, "qc_score is at least 0.85"),
        (">", 0.85, "qc_score is strictly above 0.85"),
        ("<=", 0.85, "qc_score is at most 0.85"),
        ("<", 0.85, "qc_score is strictly below 0.85"),
    ],
)
def test_a_cutoff_says_whether_the_boundary_counts(op, value, phrase):
    assert phrase in render_mission(spec(filters=[Filter(column="qc_score", op=op, value=value)]))


def test_categorical_filters_are_stated():
    text = render_mission(
        spec(
            filters=[
                Filter(column="protein", op="==", value="NEFL"),
                Filter(column="disease", op="in", value=["AD", "PD"]),
            ]
        )
    )

    assert "protein is NEFL" in text
    assert "disease is one of AD, PD" in text


def test_no_filter_means_no_scope_sentence():
    assert render_mission(spec(filters=[])).startswith("For each disease group")


def test_two_grouping_columns_are_both_named():
    text = render_mission(spec(group_by=["disease", "protein"], agg="mean", reduce="idxmax"))

    assert "combination of disease and protein" in text


def test_the_text_never_names_a_pandas_function():
    for agg, reduce in COMBOS:
        text = render_mission(spec(agg=agg, reduce=reduce)).lower()

        for word in ("groupby", "idxmax", "idxmin", "nlargest", "agg(", ".apply", "diff_extremes"):
            assert word not in text, (agg, reduce, word)


# ------------------------------------------------------------- context, topics


def test_context_opens_the_question():
    text = render_mission(spec(), context="You are comparing disease cohorts in a proteomics panel.")

    assert text.startswith("You are comparing disease cohorts in a proteomics panel. Use only rows")


def test_a_cleaning_topic_names_the_defects_at_level_one_only():
    kinds = TOPIC_MESS["missing_duplicates"]

    first = render_mission(spec(), kinds=kinds, level=1)
    later = render_mission(spec(), kinds=kinds, level=3)

    assert "-999" in first
    assert "appear twice" in first
    assert "-999" not in later
    assert "needs cleaning" in later


def test_render_challenge_replaces_what_the_model_wrote():
    model_wrote = GeneratedChallenge(
        title="Which disease group has the widest intensity spread?",
        mission="find the spread",
        spec=spec(),
        topic="groupby",
        level=2,
    )

    out = render_challenge(model_wrote)

    assert "spread" not in out.mission
    assert not out.title.lower().startswith("which")
    assert out.spec == model_wrote.spec


def test_context_with_a_number_is_rejected():
    df = generate_biomarker_dataset(n=5_000, seed=1)
    challenge = render_challenge(
        GeneratedChallenge(
            title="x",
            context="Look only at readings above 0.9.",
            spec=spec(),
            topic="groupby",
            level=2,
        )
    )

    with pytest.raises(Rejection) as exc:
        validate(challenge, df)

    assert "context" in exc.value.reason


@pytest.mark.parametrize("group_by", [["disease"], ["protein"], ["disease", "protein"]])
@pytest.mark.parametrize("agg,reduce", COMBOS)
def test_every_title_fits_without_being_cut(agg, reduce, group_by):
    title = render_title(spec(agg=agg, reduce=reduce, group_by=group_by))

    assert 0 < len(title) <= 60
    # never sliced: the title is always one of the complete wordings
    from morning_drill.render import _title_forms

    assert title in list(_title_forms(spec(agg=agg, reduce=reduce, group_by=group_by)))
