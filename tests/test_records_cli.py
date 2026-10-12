"""
Correcting the record after the fact.
"""

import json

import pytest

from morning_drill import records
from morning_drill.drill_session import Session
from scripts import records as cli


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("DRILL_DATA_DIR", str(tmp_path))
    (tmp_path / "logs").mkdir()
    return tmp_path


def failing_session(data, misses=3):
    """A session that spends `misses` wrong attempts on one question."""
    session = Session(
        progress_path=data / "progress.json",
        log_path=data / "logs" / "session.jsonl",
        records_path=data / "drill_records.json",
        agents=False,
        auto_next=False,
    )
    session.start()

    for _ in range(misses):
        session.submit("not the answer", code="answer = df.head()")

    return session


def test_list_shows_what_was_attempted_and_what_the_grader_blamed(data, capsys):
    failing_session(data)

    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out

    assert "3 att" in out
    assert "FAIL" in out
    assert "grader said:" in out


def test_flagging_a_past_question_removes_its_misses_from_what_the_agent_reads(data, capsys):
    failing_session(data)
    path = data / "drill_records.json"

    before = records.digest(path)
    topic = next(iter(before["topics"]))
    assert before["topics"][topic]["recent_misses"]
    assert before["topics"][topic]["recurring_errors"]

    assert cli.main(["flag", "1", "the", "question", "was", "unanswerable"]) == 0

    after = records.digest(path)

    assert after["topics"] == {}
    assert "taken back" in capsys.readouterr().out


def test_flagging_takes_the_attempts_back_out_of_progress(data):
    failing_session(data)

    before = json.loads((data / "progress.json").read_text())["python"]
    topic = next(t for t, s in before.items() if s.get("attempts"))
    spent = before[topic]["attempts"]
    assert spent == 3

    cli.main(["flag", "1", "wrong"])

    after = json.loads((data / "progress.json").read_text())["python"][topic]

    assert after["attempts"] == 0 and after["correct"] == 0


def test_the_flag_and_its_reason_are_stored_and_the_question_is_not_dealt_again(data):
    failing_session(data)
    path = data / "drill_records.json"
    challenge_id = records.challenges(path)[0][1]["challenge_id"]

    cli.main(["flag", challenge_id, "ambiguous", "wording"])

    stored = records.challenges(path)[0][1]
    digest = records.digest(path)

    assert stored["flagged"] is True
    assert stored["flag_reason"] == "ambiguous wording"
    assert challenge_id in digest["flagged_ids"]


def test_flagging_the_same_one_twice_is_harmless(data, capsys):
    failing_session(data)

    cli.main(["flag", "1", "wrong"])
    capsys.readouterr()

    assert cli.main(["flag", "1", "wrong"]) == 0
    assert "already flagged" in capsys.readouterr().out
    assert json.loads((data / "progress.json").read_text())["python"]


def test_an_unknown_challenge_is_an_error(data, capsys):
    failing_session(data)

    assert cli.main(["flag", "99"]) == 1
    assert cli.main(["flag", "gen_nope_00000000"]) == 1


def test_level_sets_the_level_and_optionally_the_status(data, capsys):
    (data / "progress.json").write_text(
        json.dumps({"python": {"groupby": {"status": "needs_rework", "level": 1}}})
    )

    assert cli.main(["level", "groupby", "3", "active"]) == 0

    state = json.loads((data / "progress.json").read_text())["python"]["groupby"]
    assert (state["level"], state["status"]) == (3, "active")
    assert "level 1 -> active level 3" in capsys.readouterr().out


def test_level_rejects_a_bad_status(data):
    (data / "progress.json").write_text(json.dumps({"python": {"groupby": {"level": 1}}}))

    assert cli.main(["level", "groupby", "2", "bogus"]) == 1
    assert json.loads((data / "progress.json").read_text())["python"]["groupby"]["level"] == 1
