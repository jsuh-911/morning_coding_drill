"""
The session graph: interrupt/resume, the retry loop, and progression.
"""

import json

import pandas as pd
import pytest

from morning_drill.agents.schemas import GeneratedChallenge
from morning_drill.drill_session import Session
from morning_drill.spec import ChallengeSpec, evaluate_spec


@pytest.fixture
def session(tmp_path):
    """
    Every path is redirected into tmp_path. A fixture that leaves
    records_path unset reads and writes the project's real
    drill_records.json, which both pollutes it and makes the progress
    agent's decisions depend on whatever happens to be in it.
    """
    return Session(
        progress_path=tmp_path / "progress.json",
        log_path=tmp_path / "session.jsonl",
        records_path=tmp_path / "drill_records.json",
        agents=False,
        auto_next=False,
    )


def correct_answer(session):
    spec = ChallengeSpec(**session.challenge["spec"])
    return evaluate_spec(session.data, spec)


def reference_code(session):
    return session._last_state["reference_solution"]


def test_start_pauses_for_the_student(session):
    df = session.start()

    assert isinstance(df, pd.DataFrame)
    assert session.challenge["mission"]
    # nothing graded yet
    assert session._last_state.get("correct") is None


def test_a_correct_submission_passes(session):
    session.start()
    assert session.submit(correct_answer(session), code=reference_code(session))


def test_the_retry_loop_stays_on_the_same_challenge(session):
    session.start()
    first_id = GeneratedChallenge(**session.challenge).id

    session.submit("PD", code="answer = df.head()")
    session.submit("ALS", code="answer = df.head()")

    assert GeneratedChallenge(**session.challenge).id == first_id
    assert session._last_state["attempt"] == 2


def test_attempt_number_increments(session):
    session.start()

    for expected in (1, 2, 3):
        session.submit("PD", code="answer = df.head()")
        assert session._last_state["attempt"] == expected


def test_submitting_before_starting_is_an_error(session):
    with pytest.raises(RuntimeError):
        session.submit("AD")


def test_progress_is_written(session, tmp_path):
    session.start()
    session.submit("PD", code="answer = df.head()")
    session.submit(correct_answer(session), code=reference_code(session))

    state = json.loads((tmp_path / "progress.json").read_text())
    groupby = state["python"]["groupby"]

    assert groupby["attempts"] == 2
    assert groupby["correct"] == 1


def test_every_attempt_is_logged(session, tmp_path):
    session.start()
    session.submit("PD", code="answer = df.head()")
    session.submit(correct_answer(session), code=reference_code(session))

    lines = (tmp_path / "session.jsonl").read_text().strip().splitlines()
    records = [json.loads(line) for line in lines]

    assert len(records) == 2
    assert [r["passed"] for r in records] == [False, True]
    assert records[0]["error_class"]


def test_a_clean_pass_raises_the_level_on_the_next_challenge(session, tmp_path):
    session.start()
    assert session.challenge["level"] == 1
    session.submit(correct_answer(session), code=reference_code(session))

    session.start()

    state = json.loads((tmp_path / "progress.json").read_text())
    assert state["python"]["groupby"]["level"] == 2
    assert session.challenge["level"] == 2


def test_a_hard_won_pass_does_not_raise_the_level(session, tmp_path):
    session.start()

    for _ in range(3):
        session.submit("PD", code="answer = df.head()")

    session.submit(correct_answer(session), code=reference_code(session))
    session.start()

    state = json.loads((tmp_path / "progress.json").read_text())
    assert state["python"]["groupby"]["status"] == "needs_rework"
    assert state["python"]["groupby"]["level"] == 1


def test_challenges_do_not_repeat_within_a_session(session):
    seen = []

    for _ in range(3):
        session.start()
        seen.append(GeneratedChallenge(**session.challenge).id)
        session.submit(correct_answer(session), code=reference_code(session))

    assert len(set(seen)) == 3


def test_a_typed_in_answer_does_not_pass(session):
    session.start()
    answer = correct_answer(session)

    assert not session.submit(answer, code=f"answer = {answer!r}")
    assert any("literal" in v for v in session._last_state["violations"])


def test_a_looped_solution_does_not_pass(session):
    session.start()

    assert not session.submit(
        correct_answer(session),
        code="answer = None\nfor d in df.disease.unique():\n    answer = d",
    )


def test_the_reference_is_withheld_until_the_challenge_is_solved(session, capsys):
    session.start()
    session.submit("PD", code="answer = df.head()")

    assert session.reference() is None
    assert "Finish the challenge first" in capsys.readouterr().out

    session.submit(correct_answer(session), code=reference_code(session))
    assert session.reference()


def test_the_reference_solution_actually_solves_it(session):
    """
    The student is shown code generated from the spec; running it must pass.
    """
    session.start()

    namespace = {"df": session.data, "pd": pd}
    exec(reference_code(session), namespace)  # noqa: S102

    assert session.submit(namespace["answer"], code=reference_code(session))


def test_auto_next_deals_the_following_challenge(tmp_path):
    session = Session(
        progress_path=tmp_path / "progress.json",
        log_path=tmp_path / "session.jsonl",
        records_path=tmp_path / "drill_records.json",
        agents=False,
        auto_next=True,
    )
    session.start()
    first_id = GeneratedChallenge(**session.challenge).id

    session.submit(correct_answer(session), code=reference_code(session))

    assert GeneratedChallenge(**session.challenge).id != first_id
    assert session._last_state.get("correct") is None  # the new one is unanswered


def test_every_dealt_challenge_is_discriminating(session):
    for _ in range(4):
        session.start()
        report = session._last_state["discrimination"]

        if session.challenge["title"] != "Highest NEFL group":
            assert report["discriminating"], session.challenge["title"]

        session.submit(correct_answer(session), code=reference_code(session))
