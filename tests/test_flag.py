"""
Flagging a question as wrong: it counts for nothing, is never dealt again,
and a new one takes its place.
"""

import json

import pytest

from morning_drill import records
from morning_drill.agents.schemas import GeneratedChallenge
from morning_drill.drill_session import Session
from morning_drill.spec import ChallengeSpec, evaluate_spec


def make_session(tmp_path):
    return Session(
        progress_path=tmp_path / "progress.json",
        log_path=tmp_path / "session.jsonl",
        records_path=tmp_path / "drill_records.json",
        agents=False,
        auto_next=False,
    )


@pytest.fixture
def session(tmp_path):
    return make_session(tmp_path)


def challenge_id(session):
    return GeneratedChallenge(**session.challenge).id


def topic_counts(session):
    topic = session.challenge["topic"]
    skill = session.progress["python"][topic]
    return skill.get("attempts", 0), skill.get("correct", 0)


def test_flagging_deals_a_different_question(session):
    session.start()
    first = challenge_id(session)

    session.flag("the mission asks for a protein but wants a number")

    assert challenge_id(session) != first


def test_a_flagged_question_never_comes_back_even_when_the_library_recycles(session):
    from morning_drill.agents.library import LIBRARY

    session.start()
    flagged = set()

    # flag all but one: each replacement is a question not flagged before
    for _ in range(len(LIBRARY) - 1):
        flagged.add(challenge_id(session))
        session.flag("wrong")
        assert challenge_id(session) not in flagged


def test_flagging_the_whole_library_says_so_instead_of_redealing(session):
    from morning_drill.agents.library import LIBRARY

    session.start()

    for _ in range(len(LIBRARY) - 1):
        session.flag("wrong")

    with pytest.raises(RuntimeError, match="flagged"):
        session.flag("wrong")


def test_flagged_attempts_are_taken_back_out_of_progress(session):
    session.start()
    before = topic_counts(session)

    session.submit("PD", code="answer = df.head()")
    session.submit("ALS", code="answer = df.head()")
    assert topic_counts(session)[0] == before[0] + 2

    session.flag("unanswerable")

    assert topic_counts(session) == before


def test_a_flagged_pass_does_not_count_as_correct(session):
    session.start()
    before = topic_counts(session)
    spec = ChallengeSpec(**session.challenge["spec"])

    assert session.submit(
        evaluate_spec(session.data, spec),
        code=session._last_state["reference_solution"],
    )
    assert topic_counts(session) == (before[0] + 1, before[1] + 1)

    session.flag("passed, but the question was wrong")

    assert topic_counts(session) == before


def test_the_flag_is_written_to_the_record_with_its_reason(session, tmp_path):
    session.start()
    cid = challenge_id(session)

    session.flag("which protein, but returns a number")

    data = records.load(tmp_path / "drill_records.json")
    flagged = [
        c
        for s in data["sessions"]
        for c in s["challenges"]
        if c.get("flagged")
    ]

    assert [c["challenge_id"] for c in flagged] == [cid]
    assert flagged[0]["flag_reason"] == "which protein, but returns a number"


def test_flagged_questions_stay_out_of_the_scores(session, tmp_path):
    session.start()
    session.submit("PD", code="answer = df.head()")
    session.flag("wrong")

    data = records.load(tmp_path / "drill_records.json")

    assert data["sessions"][0]["score"]["challenges"] == 0
    assert records.digest(tmp_path / "drill_records.json")["topics"] == {}


def test_tomorrows_session_does_not_deal_a_flagged_question(tmp_path):
    first = make_session(tmp_path)
    first.start()
    cid = challenge_id(first)
    signature = GeneratedChallenge(**first.challenge).spec_signature
    first.flag("wrong")

    digest = records.digest(tmp_path / "drill_records.json")
    assert cid in digest["flagged_ids"]
    assert signature in digest["avoid"]

    tomorrow = make_session(tmp_path)
    tomorrow.start()

    assert challenge_id(tomorrow) != cid


def test_flag_does_not_apply_the_previous_outcome_twice(session):
    def answer():
        return evaluate_spec(session.data, ChallengeSpec(**session.challenge["spec"]))

    session.start()
    session.submit(answer(), code=session._last_state["reference_solution"])
    session.start()  # applies the first outcome

    level = session.progress["python"][session.challenge["topic"]]["level"]

    session.flag("wrong")

    assert session._outcome is None
    assert session.progress["python"][session.challenge["topic"]]["level"] == level


def test_flagging_before_anything_is_dealt_is_refused(session):
    with pytest.raises(RuntimeError):
        session.flag("nothing yet")


def test_the_flag_is_logged(session, tmp_path):
    session.start()
    session.flag("because")

    lines = [json.loads(l) for l in (tmp_path / "session.jsonl").read_text().splitlines()]

    assert any(l.get("reason") == "because" for l in lines)
