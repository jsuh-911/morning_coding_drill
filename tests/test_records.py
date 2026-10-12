"""
drill_records.json: scoring, the digest, and the cross-day behaviour that the
whole adaptive loop rests on.
"""

import json
from datetime import date, timedelta

import pytest

from morning_drill import records
from morning_drill.agents.library import LIBRARY
from morning_drill.agents.progress_agent import decide
from morning_drill.agents.schemas import Brief


def challenge(topic="groupby", level=1, attempts=1, passed=True, **kw):
    base = {
        "challenge_id": f"gen_{topic}_{level}{attempts}{int(passed)}",
        "spec_signature": kw.pop("signature", f"{topic}|sig|{level}"),
        "topic": topic,
        "level": level,
        "attempts": attempts,
        "passed": passed,
        "error_classes": kw.pop("error_classes", []),
        "root_causes": kw.pop("root_causes", []),
        "violations": [],
        "hints_used": 0,
    }
    base.update(kw)
    return base


def session(day, challenges, notes=""):
    return {
        "session_id": f"{day}T07:00:00",
        "date": day,
        "notes": notes,
        "challenges": challenges,
    }


def days_ago(n):
    return (date.today() - timedelta(days=n)).isoformat()


# -------------------------------------------------------------------- scoring


@pytest.mark.parametrize(
    "attempts,expected", [(1, 1.0), (2, 0.6), (3, 0.3), (4, 0.1), (12, 0.1)]
)
def test_score_decreases_with_attempts(attempts, expected):
    assert records.challenge_score(attempts, True) == expected


def test_an_unfinished_challenge_scores_zero():
    assert records.challenge_score(1, False) == 0.0


def test_session_score_summarises():
    score = records.session_score(
        [
            challenge(attempts=1),
            challenge(attempts=3),
            challenge(attempts=2, passed=False),
        ]
    )

    assert score == {
        "challenges": 3,
        "passed": 2,
        "first_try": 1,
        "mean_attempts": 2.0,
        "points": round((1.0 + 0.3 + 0.0) / 3, 2),
    }


def test_session_score_of_nothing_is_zero():
    assert records.session_score([])["points"] == 0.0


# ---------------------------------------------------------------- the file


def test_missing_file_reads_as_empty(tmp_path):
    assert records.load(tmp_path / "nope.json") == records.empty()


def test_corrupt_file_reads_as_empty_rather_than_raising(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("{ this is not json")

    assert records.load(path) == records.empty()


def test_round_trip(tmp_path):
    path = tmp_path / "r.json"
    records.upsert_session(path, session(days_ago(1), [challenge()]))

    data = records.load(path)

    assert data["schema"] == records.SCHEMA
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["score"]["points"] == 1.0


def test_upsert_replaces_the_same_session_rather_than_duplicating(tmp_path):
    path = tmp_path / "r.json"
    day = days_ago(0)

    records.upsert_session(path, session(day, [challenge(attempts=1)]))
    records.upsert_session(
        path, session(day, [challenge(attempts=1), challenge(topic="groupby", attempts=2)])
    )

    data = records.load(path)

    assert len(data["sessions"]) == 1
    assert len(data["sessions"][0]["challenges"]) == 2


def test_two_sessions_in_one_day_are_both_kept(tmp_path):
    path = tmp_path / "r.json"
    day = days_ago(0)

    morning = session(day, [challenge()])
    evening = session(day, [challenge(attempts=2)])
    evening["session_id"] = f"{day}T19:00:00"

    records.upsert_session(path, morning)
    records.upsert_session(path, evening)

    assert len(records.load(path)["sessions"]) == 2


def test_an_unattempted_challenge_is_not_written(tmp_path):
    path = tmp_path / "r.json"
    records.upsert_session(
        path, session(days_ago(0), [challenge(attempts=0, passed=False)])
    )

    assert records.load(path)["sessions"] == []


# ----------------------------------------------------------------- digest


def test_digest_aggregates_per_topic(tmp_path):
    path = tmp_path / "r.json"
    records.upsert_session(
        path,
        session(
            days_ago(2),
            [
                challenge(attempts=1, error_classes=["correct"]),
                challenge(attempts=3, error_classes=["wrong_shape", "conceptual"]),
                challenge(topic="dataframe_filtering", attempts=1),
            ],
        ),
    )

    d = records.digest(path)

    assert d["topics"]["groupby"]["challenges"] == 2
    assert d["topics"]["groupby"]["mean_attempts"] == 2.0
    assert d["topics"]["groupby"]["pass_rate"] == 1.0
    assert d["topics"]["groupby"]["days_since"] == 2
    assert "wrong_shape" in d["topics"]["groupby"]["recurring_errors"]


def test_digest_collects_signatures_to_avoid(tmp_path):
    path = tmp_path / "r.json"
    records.upsert_session(
        path,
        session(
            days_ago(1),
            [challenge(signature="sig-a"), challenge(signature="sig-b")],
        ),
    )

    assert records.digest(path)["avoid"] == ["sig-a", "sig-b"]


def test_digest_carries_the_learners_own_notes(tmp_path):
    path = tmp_path / "r.json"
    records.upsert_session(
        path, session(days_ago(1), [challenge()], notes="nlargest felt shaky")
    )

    text = records.digest_text(records.digest(path))

    assert "nlargest felt shaky" in text


def test_digest_excludes_submitted_code_from_the_prompt(tmp_path):
    """
    Code is kept in the file, but months of source must not reach a prompt.
    """
    path = tmp_path / "r.json"
    records.upsert_session(
        path,
        session(
            days_ago(1),
            [
                challenge(
                    attempts_log=[
                        {"attempt": 1, "code": "SECRET_MARKER = 1", "passed": True}
                    ]
                )
            ],
        ),
    )

    assert "SECRET_MARKER" in (tmp_path / "r.json").read_text()
    assert "SECRET_MARKER" not in json.dumps(records.digest(path))
    assert "SECRET_MARKER" not in records.digest_text(records.digest(path))


def test_digest_of_an_empty_record_reads_as_a_first_session(tmp_path):
    assert "first one" in records.digest_text(records.digest(tmp_path / "r.json"))


# ------------------------------------------------------- delayed retrieval


MASTERED = ["dataframe_filtering", "groupby"]


def test_a_stale_mastered_topic_is_due(tmp_path):
    path = tmp_path / "r.json"
    records.upsert_session(
        path, session(days_ago(30), [challenge(topic="dataframe_filtering", level=4)])
    )

    assert records.retrieval_due(records.digest(path), MASTERED) == "dataframe_filtering"


def test_a_freshly_practised_topic_is_not_due(tmp_path):
    path = tmp_path / "r.json"
    records.upsert_session(
        path, session(days_ago(1), [challenge(topic="dataframe_filtering", level=4)])
    )

    assert records.retrieval_due(records.digest(path), MASTERED) is None


def test_the_stalest_topic_wins(tmp_path):
    path = tmp_path / "r.json"
    records.upsert_session(
        path, session(days_ago(40), [challenge(topic="dataframe_filtering", level=4)])
    )
    records.upsert_session(
        path, session(days_ago(10), [challenge(topic="groupby", level=4)])
    )

    assert records.retrieval_due(records.digest(path), MASTERED) == "dataframe_filtering"


def test_mastery_claimed_but_never_observed_waits(tmp_path):
    """
    progress.json ships with three topics marked mastered that this drill has
    never watched anyone pass. Those claims must not trigger level 4 on day
    one -- only once there is enough record to audit against.
    """
    path = tmp_path / "r.json"
    records.upsert_session(path, session(days_ago(1), [challenge(topic="groupby")]))

    assert records.retrieval_due(records.digest(path), ["dataframe_filtering"]) is None

    for day in (5, 4, 3):
        records.upsert_session(
            path, session(days_ago(day), [challenge(topic="groupby")])
        )

    d = records.digest(path)
    assert d["sessions"] >= records.UNVERIFIED_AUDIT_AFTER_SESSIONS
    assert records.retrieval_due(d, ["dataframe_filtering"]) == "dataframe_filtering"


def test_the_excluded_topic_is_never_chosen(tmp_path):
    path = tmp_path / "r.json"
    records.upsert_session(
        path, session(days_ago(30), [challenge(topic="dataframe_filtering", level=4)])
    )

    due = records.retrieval_due(
        records.digest(path), MASTERED, exclude=("dataframe_filtering",)
    )

    assert due != "dataframe_filtering"


# --------------------------------------------- the progress agent uses it


PROGRESS = {
    "python": {
        "dataframe_filtering": {"status": "mastered", "level": 4},
        "groupby": {"status": "active", "level": 2, "attempts": 4, "correct": 3},
    }
}


def test_the_progress_agent_schedules_stale_retrieval_without_a_model(tmp_path):
    path = tmp_path / "r.json"
    records.upsert_session(
        path, session(days_ago(21), [challenge(topic="dataframe_filtering", level=4)])
    )

    decision = decide(
        None,
        PROGRESS,
        outcome={"topic": "groupby", "level": 2, "passed": True, "attempt": 1},
        digest=records.digest(path),
    )

    assert decision.brief.topic == "dataframe_filtering"
    assert decision.brief.level == 4
    assert "groupby" in decision.brief.fold_in


def test_avoid_signatures_reach_the_brief():
    decision = decide(None, PROGRESS, avoid=["sig-a", "sig-b"])
    assert decision.brief.avoid == ["sig-a", "sig-b"]


def test_the_trainer_agent_is_told_what_not_to_repeat(tmp_path):
    """
    The point of the whole record: yesterday's question is not re-asked today.
    """
    from morning_drill.agents import prompts

    brief = Brief(
        topic="groupby",
        level=2,
        rationale="x",
        avoid=[LIBRARY[0].spec_signature],
    )

    prompt = prompts.TRAINER_PROMPT.format(
        rules="", dataset="", domain="python", topic=brief.topic,
        level=brief.level, rationale=brief.rationale,
        fold_in="nothing", avoid=", ".join(brief.avoid), data_notes="",
    )

    assert LIBRARY[0].spec_signature in prompt


def test_challenge_ids_are_stable_across_processes():
    """
    Regression: the id used hash(), which is salted per interpreter, so
    nothing recorded yesterday ever matched anything generated today.
    """
    import subprocess
    import sys

    from conftest import ROOT

    ids = [
        subprocess.run(
            [sys.executable, "-c",
             "from morning_drill.agents.library import LIBRARY; print(LIBRARY[0].id)"],
            capture_output=True, text=True, cwd=ROOT,
            env={"PYTHONPATH": str(ROOT), "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "12345")
    ]

    assert ids[0] and ids[0] == ids[1], ids


def test_only_failures_count_as_mistakes(tmp_path):
    """
    A passing review has error_class "correct" and a root_cause describing
    what went right. Recording those made the digest report "recurring:
    correct x1" and quote the passing review as the reason for a miss.
    """
    from morning_drill.drill_session import Session
    from morning_drill.spec import ChallengeSpec, evaluate_spec

    s = Session(
        progress_path=tmp_path / "p.json",
        log_path=tmp_path / "l.jsonl",
        records_path=tmp_path / "r.json",
        agents=False,
        auto_next=False,
    )
    s.start()
    s.submit("PD", code="answer = df.head()")
    s.submit(
        evaluate_spec(s.data, ChallengeSpec(**s.challenge["spec"])),
        code=s._last_state["reference_solution"],
    )
    s.save()

    recorded = records.load(tmp_path / "r.json")["sessions"][0]["challenges"][0]

    assert recorded["passed"] is True
    assert recorded["attempts"] == 2
    assert "correct" not in recorded["error_classes"]
    assert len(recorded["root_causes"]) == 1          # the failed attempt only
    assert len(recorded["attempts_log"]) == 2         # both attempts' code kept


# ------------------------------------------------ only topics we can set


def test_a_topic_with_no_generator_is_never_set():
    """
    The curriculum lists SQL and the command line, but the trainer can only
    write questions on the protein table. A brief for sql/joins produced a
    pandas question under an SQL heading, with the attempts recorded against a
    topic that was never practised.
    """
    from morning_drill.agents.progress_agent import decide
    from morning_drill.agents.schemas import ProgressDecision

    class PicksSql:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            return ProgressDecision(
                topic_status="active",
                next_level=1,
                note="x",
                brief=Brief(domain="sql", topic="joins", level=1, rationale="r"),
            )

    decision = decide(PicksSql(), {"python": {}}, avoid=["sig"])

    assert (decision.brief.domain, decision.brief.topic) == ("python", "dataframe_filtering")
    assert decision.brief.avoid == ["sig"]


def test_a_python_topic_with_a_generator_is_kept():
    from morning_drill.agents.progress_agent import decide
    from morning_drill.agents.schemas import ProgressDecision

    class PicksCleaning:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            return ProgressDecision(
                topic_status="active",
                next_level=2,
                note="x",
                brief=Brief(topic="string_ops", level=2, rationale="r"),
            )

    assert decide(PicksCleaning(), {"python": {}}).brief.topic == "string_ops"


def test_mastering_a_topic_moves_on_to_the_next_one():
    """
    After groupby survived level 4 the rule-based brief was groupby level 4
    again, so the cleaning topics after it were never reached.
    """
    from morning_drill.agents.progress_agent import fallback_decision

    progress = {
        "python": {
            "dataframe_filtering": {"status": "mastered", "level": 4},
            "groupby": {"status": "active", "level": 4},
        }
    }
    outcome = {"topic": "groupby", "level": 4, "passed": True, "attempt": 1, "domain": "python"}

    decision = fallback_decision(progress, outcome, history=[{"passed": True}], digest=None)

    assert decision.topic_status == "mastered"
    assert decision.brief.topic == "missing_duplicates"
    assert decision.brief.level == 1


def test_a_topic_that_is_not_mastered_stays_put():
    from morning_drill.agents.progress_agent import fallback_decision

    progress = {"python": {"groupby": {"status": "active", "level": 2}}}
    outcome = {"topic": "groupby", "level": 2, "passed": True, "attempt": 1, "domain": "python"}

    decision = fallback_decision(progress, outcome, history=[{"passed": True}], digest=None)

    assert decision.brief.topic == "groupby"
    assert decision.brief.level == 3


def test_the_progress_agent_is_only_shown_topics_it_can_set():
    from morning_drill.agents.progress_agent import _format_progress

    text = _format_progress(
        {
            "python": {"groupby": {"status": "active", "level": 2}},
            "sql": {"joins": {"status": "active", "level": 1}},
            "cli": {"file_operations": {"status": "mastered", "level": 4}},
        }
    )

    assert "groupby" in text
    assert "joins" not in text and "file_operations" not in text
    assert "sql" not in text and "cli" not in text
