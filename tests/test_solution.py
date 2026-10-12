"""
Showing the solution ends the challenge as a miss.
"""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from morning_drill import records
from morning_drill.agents.schemas import GeneratedChallenge
from morning_drill.drill_session import Session
from morning_drill.spec import ChallengeSpec, evaluate_spec
from web import app as web_app


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


def counts(session):
    topic = session.challenge["topic"]
    skill = session.progress["python"][topic]
    return skill.get("attempts", 0), skill.get("correct", 0)


def recorded(tmp_path):
    return records.challenges(tmp_path / "drill_records.json")[0][1]


def test_it_shows_the_reference_solution_and_the_expected_answer(session):
    session.start()

    shown = session.give_up()

    assert shown["reference"] == session._last_state["reference_solution"]
    assert shown["expected"] == evaluate_spec(
        session.data, ChallengeSpec(**session.challenge["spec"])
    )
    assert session.gave_up


def test_it_counts_as_a_miss_not_a_pass(session, tmp_path):
    session.start()
    before = counts(session)

    session.give_up()

    assert counts(session) == (before[0] + 1, before[1])

    c = recorded(tmp_path)
    assert c["passed"] is False
    assert c["gave_up"] is True
    assert c["attempts"] == 1
    assert records.challenge_score(c["attempts"], c["passed"]) == 0.0


def test_the_progress_agent_is_told_why(session, tmp_path):
    session.start()
    session.give_up()

    topic = next(iter(records.digest(tmp_path / "drill_records.json")["topics"].values()))

    assert topic["recent_misses"][0]["why"] == "gave up and looked at the solution"
    assert topic["recurring_errors"] == {"gave_up": 1}


def test_attempts_already_spent_stay_and_the_give_up_adds_one(session, tmp_path):
    session.start()
    session.submit("wrong", code="answer = df.head()")
    session.submit("wrong", code="answer = df.head()")

    session.give_up()

    assert recorded(tmp_path)["attempts"] == 3


def test_it_is_idempotent(session, tmp_path):
    session.start()

    first = session.give_up()
    before = counts(session)
    second = session.give_up()

    assert counts(session) == before
    assert second["reference"] == first["reference"]
    assert recorded(tmp_path)["attempts"] == 1


def test_submitting_after_giving_up_is_refused(session):
    session.start()
    session.give_up()

    with pytest.raises(RuntimeError, match="solution"):
        session.submit("anything", code="answer = 1")


def test_the_next_challenge_starts_clean(session):
    session.start()
    first = GeneratedChallenge(**session.challenge).id
    session.give_up()

    session.start()

    assert not session.gave_up
    assert GeneratedChallenge(**session.challenge).id != first
    assert session.submit("PD", code="answer = df.head()") is False  # accepts a submission again


def test_it_needs_a_challenge(session):
    with pytest.raises(RuntimeError):
        session.give_up()


def test_flagging_after_giving_up_gives_the_attempt_back(session):
    session.start()
    before = counts(session)

    session.give_up()
    session.flag("the question was wrong")

    assert counts(session)[0] == before[0]


def test_a_cleaning_solution_includes_the_cleaning(tmp_path):
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
    session = make_session(tmp_path)
    session.start()

    shown = session.give_up()

    assert session.box.truth is not None
    assert "df = df.drop_duplicates()" in shown["reference"] or "replace(" in shown["reference"]


# ------------------------------------------------------------------- the web


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DRILL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DRILL_AGENTS", "0")
    monkeypatch.delenv("DRILL_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("DRILL_EXEC_MODE", "inprocess")
    (tmp_path / "logs").mkdir(exist_ok=True)
    monkeypatch.setattr(web_app, "_session", None)
    return TestClient(web_app.app)


def test_the_endpoint_returns_the_solution_and_the_expected_answer(client):
    client.post("/api/start", json={})

    body = client.post("/api/solution", json={}).json()

    assert "answer" in body["solution"]["reference"]
    assert body["solution"]["expected"]["repr"]
    assert body["gave_up"] is True


def test_submitting_after_the_solution_is_a_409_with_a_reason(client):
    client.post("/api/start", json={})
    client.post("/api/solution", json={})

    response = client.post("/api/submit", json={"code": "answer = 1"})

    assert response.status_code == 409
    assert "solution" in response.json()["detail"]


def test_the_state_remembers_it_across_a_reload(client):
    client.post("/api/start", json={})
    client.post("/api/solution", json={})

    assert client.get("/api/state").json()["gave_up"] is True


def test_a_new_challenge_clears_it(client):
    client.post("/api/start", json={})
    client.post("/api/solution", json={})

    body = client.post("/api/start", json={}).json()

    assert body["gave_up"] is False
    assert client.post("/api/submit", json={"code": "answer = 1"}).status_code == 200


def test_the_solution_before_any_challenge_is_a_409(client):
    assert client.post("/api/solution", json={}).status_code == 409
