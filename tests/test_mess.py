"""
Cleaning topics: the student is handed the raw table, the answer is computed
on the clean one, and the topic alone decides which they get.
"""

import json

import numpy as np
import pandas as pd
import pytest

from morning_drill.agents.library import LIBRARY
from morning_drill.agents.schemas import GeneratedChallenge
from morning_drill.agents.trainer_agent import Rejection, validate
from morning_drill.agents.grader_agent import verify
from morning_drill.drill_session import Session
from morning_drill.generators import generate_biomarker_dataset
from morning_drill.mess import (
    KINDS,
    SENTINEL,
    TOPIC_MESS,
    cleaning_code,
    mess_for_topic,
    mess_report,
    messify,
)
from morning_drill.probes import method_failures
from morning_drill.session import _with_cleaning
from morning_drill.spec import ChallengeSpec, evaluate_spec, spec_to_reference_code


@pytest.fixture(scope="module")
def clean():
    return generate_biomarker_dataset(n=20_000, seed=3)


def restore(raw, kinds):
    ns = {"df": raw.copy(), "np": np, "pd": pd}
    exec(cleaning_code(kinds), ns)
    return ns["df"].reset_index(drop=True)


# ------------------------------------------------------------- the defects


@pytest.mark.parametrize("kinds", [(k,) for k in KINDS] + [tuple(KINDS)])
def test_every_defect_is_exactly_reversible(clean, kinds):
    raw = messify(clean, kinds)

    assert restore(raw, kinds).equals(clean.reset_index(drop=True))


def test_the_raw_table_actually_looks_raw(clean):
    raw = messify(clean, tuple(KINDS))

    assert len(raw) > len(clean)
    assert (raw.intensity == SENTINEL).any()
    assert not raw.intensity.isna().any()
    assert set(raw.protein) - set(clean.protein)
    assert set(raw.disease) - set(clean.disease)


def test_messify_is_deterministic_and_leaves_the_original_alone(clean):
    before = clean.copy()

    assert messify(clean, tuple(KINDS), seed=1).equals(messify(clean, tuple(KINDS), seed=1))
    assert clean.equals(before)


def test_no_kinds_returns_the_table_unchanged(clean):
    assert messify(clean, ()) is clean


# -------------------------------------------- the topic decides, nothing else


def test_only_the_cleaning_topics_are_messy():
    assert mess_for_topic("missing_duplicates") == TOPIC_MESS["missing_duplicates"]
    assert mess_for_topic("string_ops") == TOPIC_MESS["string_ops"]
    assert mess_for_topic("groupby") == ()
    assert mess_for_topic("dataframe_filtering") == ()


# ------------------------------------------------- a defect that must matter


def test_a_defect_that_does_not_change_the_answer_is_rejected(clean):
    """
    Grouped by protein with no disease filter, stray spaces in the disease
    column cannot change anything: the student could skip that cleaning step.
    """
    spec = ChallengeSpec(
        filters=[{"column": "qc_score", "op": ">=", "value": 0.8}],
        group_by=["protein"],
        value="intensity",
        agg="median",
        reduce="nlargest",
    )
    challenge = GeneratedChallenge(
        title="t",
        mission="Clean the export, then return the three proteins with the highest median intensity at qc_score >= 0.8.",
        spec=spec,
        topic="string_ops",
        level=3,
    )

    assert mess_report(clean, spec, ("protein_text", "disease_text"))["ignorable"] == ["disease_text"]

    with pytest.raises(Rejection) as exc:
        validate(challenge, clean, mess_kinds=("protein_text", "disease_text"))

    assert "disease_text" in exc.value.reason


@pytest.mark.parametrize("challenge", [c for c in LIBRARY if mess_for_topic(c.topic)], ids=lambda c: c.id)
def test_every_library_cleaning_question_needs_every_cleaning_step(clean, challenge):
    report = mess_report(clean, challenge.spec, mess_for_topic(challenge.topic))

    assert report["ignorable"] == []


# --------------------------------------------------------------- the grading


def spec_code(challenge):
    return _with_cleaning(mess_for_topic(challenge.topic), spec_to_reference_code(challenge.spec))


@pytest.mark.parametrize("challenge", [c for c in LIBRARY if mess_for_topic(c.topic)], ids=lambda c: c.id)
def test_the_reference_solution_passes_value_and_probes_on_the_raw_table(clean, challenge):
    kinds = mess_for_topic(challenge.topic)
    raw = messify(clean, kinds)
    code = spec_code(challenge)

    ns = {"df": raw.copy(), "np": np, "pd": pd}
    exec(code, ns)

    result = verify(
        challenge,
        clean,
        ns["answer"],
        code,
        prepare=lambda frame: messify(frame, kinds),
    )

    assert result["correct"] is True, result["violations"]


def test_skipping_a_cleaning_step_gives_the_wrong_value(clean):
    challenge = next(c for c in LIBRARY if c.topic == "missing_duplicates" and c.level == 1)
    kinds = mess_for_topic(challenge.topic)
    raw = messify(clean, kinds)

    skipped_dupes = (
        'df["intensity"] = df["intensity"].replace(-999.0, np.nan)\n'
        + spec_to_reference_code(challenge.spec)
    )
    skipped_sentinel = "df = df.drop_duplicates()\n" + spec_to_reference_code(challenge.spec)

    for code in (skipped_dupes, skipped_sentinel):
        ns = {"df": raw.copy(), "np": np, "pd": pd}
        exec(code, ns)

        assert not verify(challenge, clean, ns["answer"], code)["correct"], code


def test_another_valid_cleaning_route_passes(clean):
    challenge = next(c for c in LIBRARY if c.topic == "missing_duplicates" and c.level == 1)
    kinds = mess_for_topic(challenge.topic)

    code = (
        "d = df.drop_duplicates(subset='sample_id')\n"
        "d = d[d.intensity > 0]\n"
        "d = d[(d.qc_score >= 0.85) & (d.protein == 'NEFL')]\n"
        "m = d.groupby('disease').intensity.mean()\n"
        "answer = m.max() - m.min()"
    )

    ns = {"df": messify(clean, kinds), "np": np, "pd": pd}
    exec(code, ns)

    result = verify(challenge, clean, ns["answer"], code, prepare=lambda f: messify(f, kinds))

    assert result["correct"] is True, result["violations"]


def test_planted_probe_rows_do_not_share_sample_ids(clean):
    from morning_drill.probes import probe_frames

    challenge = next(c for c in LIBRARY if c.topic == "missing_duplicates" and c.level == 1)

    for _, _, frame in probe_frames(clean, challenge.spec):
        assert frame.sample_id.is_unique


def test_scalar_answers_tolerate_the_last_bit():
    from morning_drill.graders import compare_answers

    assert compare_answers(1.0 + 1e-13, 1.0)
    assert not compare_answers(1.001, 1.0)


# ------------------------------------------------------------ in a session


def cleaning_session(tmp_path):
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "python": {
                    "dataframe_filtering": {"status": "mastered", "level": 4},
                    "groupby": {"status": "mastered", "level": 4},
                }
            }
        )
    )

    return Session(
        progress_path=tmp_path / "progress.json",
        log_path=tmp_path / "session.jsonl",
        records_path=tmp_path / "drill_records.json",
        agents=False,
        auto_next=False,
    )


def test_reaching_a_cleaning_topic_hands_the_student_the_raw_table(tmp_path):
    session = cleaning_session(tmp_path)
    session.start()

    assert session.challenge["topic"] in TOPIC_MESS
    assert session.box.truth is not None
    assert not session.box.truth.equals(session.data)

    raw_has_mess = (
        (session.data.intensity == SENTINEL).any()
        or session.data.protein.str.strip().str.upper().ne(session.data.protein).any()
    )
    assert raw_has_mess


def test_a_clean_topic_gets_the_clean_table(tmp_path):
    session = Session(
        progress_path=tmp_path / "progress.json",
        log_path=tmp_path / "session.jsonl",
        records_path=tmp_path / "drill_records.json",
        agents=False,
        auto_next=False,
    )
    session.start()

    assert session.challenge["topic"] not in TOPIC_MESS
    assert session.box.truth is None
    assert not (session.data.intensity == SENTINEL).any()


def test_the_reference_solution_shown_after_solving_includes_the_cleaning(tmp_path):
    session = cleaning_session(tmp_path)
    session.start()

    reference = session._last_state["reference_solution"]

    assert cleaning_code(mess_for_topic(session.challenge["topic"])).splitlines()[0] in reference


def test_solving_a_cleaning_question_end_to_end(tmp_path):
    session = cleaning_session(tmp_path)
    session.start()

    code = session._last_state["reference_solution"]
    ns = {"df": session.data.copy(), "np": np, "pd": pd}
    exec(code, ns)

    assert session.submit(ns["answer"], code=code)


def test_the_answer_without_cleaning_is_marked_wrong_in_a_session(tmp_path):
    session = cleaning_session(tmp_path)
    session.start()

    uncleaned = spec_to_reference_code(ChallengeSpec(**session.challenge["spec"]))
    ns = {"df": session.data.copy(), "np": np, "pd": pd}
    exec(uncleaned, ns)

    assert not session.submit(ns["answer"], code=uncleaned)


def test_mastering_groupby_in_a_session_leads_to_a_cleaning_topic(tmp_path):
    """
    The claim that the cleaning topics arrive once groupby is mastered, tested
    end to end rather than assumed.
    """
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "python": {
                    "dataframe_filtering": {"status": "mastered", "level": 4},
                    "groupby": {"status": "active", "level": 4},
                }
            }
        )
    )
    session = Session(
        progress_path=tmp_path / "progress.json",
        log_path=tmp_path / "session.jsonl",
        records_path=tmp_path / "drill_records.json",
        agents=False,
        auto_next=False,
    )
    session.start()
    assert session.challenge["topic"] == "groupby"

    code = session._last_state["reference_solution"]
    ns = {"df": session.data.copy(), "np": np, "pd": pd}
    exec(code, ns)
    assert session.submit(ns["answer"], code=code)

    session.start()

    assert session.challenge["topic"] == "missing_duplicates"
    assert session.box.truth is not None  # the raw table
