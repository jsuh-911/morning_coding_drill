"""
The web layer: the Run/Submit split, the sandbox, and auth.

Like the rest of the suite this needs no API key -- the app is built with
agents off, so the verdict, the constraints and the structural report are
exercised and the prose is what is missing.
"""

import pathlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from morning_drill.drill_session import Session

from web import app as web_app
from web import config, views
from web.executor import run_inprocess, run_user_code


# --------------------------------------------------------------- fixtures


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    A fresh app over a throwaway data directory. Without this the app reads
    and writes the project's real drill_records.json, which pollutes it and
    makes tomorrow's progress agent reason from test runs.
    """
    monkeypatch.setenv("DRILL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DRILL_AGENTS", "0")
    monkeypatch.delenv("DRILL_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("DRILL_EXEC_MODE", "inprocess")

    (tmp_path / "logs").mkdir(exist_ok=True)

    monkeypatch.setattr(web_app, "_session", None)

    return TestClient(web_app.app)


@pytest.fixture
def started(client):
    response = client.post("/api/start", json={})
    assert response.status_code == 200
    return client


# ------------------------------------------------------------- the split


def test_start_sets_a_challenge_with_a_dataset(started):
    body = started.post("/api/start", json={}).json()
    challenge = body["challenge"]

    assert challenge["mission"]
    assert challenge["key"]
    assert challenge["dataset"]["shape"][0] > 0
    assert challenge["attempt"] == 0


def test_challenge_carries_an_id_despite_it_being_a_property(started):
    # GeneratedChallenge.id is a property, so it is absent from the dumped
    # dict; views recomputes it. Without this the browser cannot tell one
    # challenge from the next and never resets the editor.
    challenge = started.get("/api/state").json()["challenge"]

    assert challenge["id"] and challenge["id"].startswith("gen_")


def test_run_costs_no_attempt(started):
    before = started.get("/api/state").json()["challenge"]["attempt"]

    body = started.post("/api/run", json={"code": "answer = 1"}).json()

    assert body["ok"] and body["answer_bound"]
    assert started.get("/api/state").json()["challenge"]["attempt"] == before


def test_run_reports_a_traceback_without_the_apps_own_frames(started):
    body = started.post("/api/run", json={"code": "answer = df['nope']"}).json()

    assert body["ok"] is False
    assert "KeyError" in body["traceback"]
    assert "web/executor.py" not in body["traceback"]


def test_run_captures_stdout(started):
    body = started.post("/api/run", json={"code": "print('hello')\nanswer = 1"}).json()

    assert "hello" in body["stdout"]


def test_submit_counts_an_attempt(started):
    started.post("/api/submit", json={"code": "answer = -1.0"})

    assert started.get("/api/state").json()["challenge"]["attempt"] == 1


def test_crashing_submit_counts_nothing(started):
    body = started.post("/api/submit", json={"code": "answer = 1 / 0"}).json()

    # Session.submit() has nowhere to put a traceback, so a crashed run would
    # otherwise land in the record as a wrong answer with no explanation.
    assert body["rejected"] == "error"
    assert started.get("/api/state").json()["challenge"]["attempt"] == 0


def test_submit_without_binding_answer_counts_nothing(started):
    body = started.post("/api/submit", json={"code": "x = 1"}).json()

    assert body["rejected"] == "no_answer"
    assert started.get("/api/state").json()["challenge"]["attempt"] == 0


# ------------------------------------------------------------- the verdict


def test_the_reference_does_not_leak_before_a_pass(started):
    body = started.post("/api/submit", json={"code": "answer = -1.0"}).json()

    assert body["review"]["passed"] is False
    assert "reference" not in body["review"]
    assert started.get("/api/reference").status_code == 409


def test_a_loop_is_a_violation_even_when_the_answer_is_right(started, tmp_path):
    challenge = started.get("/api/state").json()["challenge"]

    if "Do not use a for-loop" not in " ".join(challenge["constraints"]):
        pytest.skip("this challenge does not forbid loops")

    code = (
        "vals = []\n"
        "for g, sub in df.groupby('disease'):\n"
        "    vals.append(sub.intensity.median())\n"
        "answer = max(vals) - min(vals)\n"
    )

    review = started.post("/api/submit", json={"code": code}).json()["review"]

    assert review["passed"] is False
    assert review["violations"]


# ------------------------------------------------------------- the sandbox


def test_subprocess_mode_returns_a_real_object(tmp_path, monkeypatch):
    monkeypatch.setenv("DRILL_EXEC_MODE", "subprocess")

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = run_user_code("answer = df.a.sum()", df)

    assert result.ok and result.answer_bound
    assert int(result.answer) == 6


def test_subprocess_mode_kills_an_unbounded_loop(monkeypatch):
    monkeypatch.setenv("DRILL_EXEC_MODE", "subprocess")
    monkeypatch.setenv("DRILL_EXEC_TIMEOUT", "3")

    result = run_user_code("while True: pass", pd.DataFrame({"a": [1]}))

    assert result.ok is False
    assert "imed out" in result.traceback


def test_subprocess_mode_refuses_an_answer_it_cannot_return(monkeypatch):
    monkeypatch.setenv("DRILL_EXEC_MODE", "subprocess")

    result = run_user_code("answer = lambda x: x", pd.DataFrame({"a": [1]}))

    assert result.ok is False
    assert result.answer_bound is False


def test_an_unknown_exec_mode_is_refused_rather_than_guessed(monkeypatch):
    monkeypatch.setenv("DRILL_EXEC_MODE", "yolo")

    with pytest.raises(ValueError):
        config.exec_mode()


# ----------------------------------------------------------------- auth


def test_the_api_is_closed_once_a_token_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("DRILL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DRILL_AGENTS", "0")
    monkeypatch.setenv("DRILL_AUTH_TOKEN", "s3cret")
    monkeypatch.setattr(web_app, "_session", None)

    (tmp_path / "logs").mkdir(exist_ok=True)

    client = TestClient(web_app.app)

    assert client.get("/api/state").status_code == 401
    assert client.get("/api/state", headers={"X-Drill-Token": "nope"}).status_code == 401
    assert client.get("/api/state", headers={"X-Drill-Token": "s3cret"}).status_code == 200
    assert client.get("/api/state?token=s3cret").status_code == 200


def test_auth_is_checked_before_the_code_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("DRILL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DRILL_AGENTS", "0")
    monkeypatch.setenv("DRILL_AUTH_TOKEN", "s3cret")
    monkeypatch.setattr(web_app, "_session", None)

    (tmp_path / "logs").mkdir(exist_ok=True)

    marker = tmp_path / "side-effect"
    client = TestClient(web_app.app)

    response = client.post(
        "/api/run",
        json={"code": f"open({str(marker)!r}, 'w').write('x')\nanswer = 1"},
    )

    assert response.status_code == 401
    assert not marker.exists()


# ----------------------------------------------------------------- views


def test_tables_survive_json_with_nan_and_numpy_scalars():
    df = pd.DataFrame({"x": [1.0, float("nan")], "s": ["a", None]})

    table = views.table_of(df)

    # NaN is not valid JSON; it has to come back as null, not as a float.
    assert table["rows"][1][0] is None
    assert table["columns"] == ["x", "s"]


def test_a_series_answer_renders_as_a_table():
    preview = views.preview_of(pd.Series([1, 2], index=["a", "b"], name="v"))

    assert preview["kind"] == "Series"
    assert preview["table"]["rows"] == [["a", 1], ["b", 2]]


def test_a_scalar_answer_has_no_table():
    preview = views.preview_of(3.5)

    assert preview["table"] is None
    assert preview["repr"] == "3.5"


# ------------------------------------------------------- the sandbox feel


def test_the_starter_template_is_valid_python():
    # It shipped as `answer = ` with an empty right-hand side, so the very
    # first Run was a SyntaxError no matter what was typed below it.
    starter = pathlib.Path("web/static/app.js").read_text()
    body = starter.split("const STARTER = (name) =>")[1].split(";")[0]

    rendered = (
        body.replace("`", "")
        .replace("${name}", "answer")
        .replace("\\n", "\n")
        .replace(" +\n", "")
    )

    compile(rendered, "<starter>", "exec")


def test_a_trailing_expression_is_echoed_like_a_notebook():
    # df.head() on the last line is the first thing anyone types in a
    # sandbox; under plain exec() the value is silently discarded.
    result = run_inprocess("df.head()", pd.DataFrame({"a": [1, 2]}))

    assert result.ok and result.has_value
    assert list(result.value["a"]) == [1, 2]


def test_a_trailing_assignment_echoes_nothing():
    result = run_inprocess("answer = 1 + 1", pd.DataFrame({"a": [1]}))

    assert result.has_value is False
    assert result.answer == 2


def test_a_syntax_error_does_not_leak_the_apps_own_frames():
    result = run_inprocess("answer = ", pd.DataFrame({"a": [1]}))

    assert result.ok is False
    assert "SyntaxError" in result.traceback
    assert "executor.py" not in result.traceback


def test_a_runtime_error_keeps_the_line_number():
    # exec() on a raw string names its frames "<string>", which the filter
    # drops -- so the failing line disappeared entirely.
    result = run_inprocess("x = 1\nanswer = df['nope']\n", pd.DataFrame({"a": [1]}))

    assert 'File "<submission>", line 2' in result.traceback
    assert "executor.py" not in result.traceback


def test_a_traceback_keeps_frames_from_the_learners_own_functions():
    result = run_inprocess(
        "def f():\n    return 1 / 0\nanswer = f()\n", pd.DataFrame({"a": [1]})
    )

    assert result.traceback.count("<submission>") == 2
    assert "ZeroDivisionError" in result.traceback


def test_the_echoed_value_reaches_the_browser(started):
    body = started.post("/api/run", json={"code": "df.head(2)"}).json()

    assert body["ok"] is True
    assert body["value"]["table"]["rows"]
    assert body["answer_bound"] is False


# ------------------------------------------------------------------ flag


def test_flag_deals_a_new_question_and_costs_nothing(started):
    first = started.get("/api/state").json()["challenge"]["id"]

    response = started.post("/api/flag", json={"reason": "unclear"})

    assert response.status_code == 200
    body = response.json()
    assert body["challenge"]["id"] != first
    assert body["challenge"]["attempt"] == 0


def test_flag_before_start_is_a_409(client):
    assert client.post("/api/flag", json={"reason": "x"}).status_code == 409


# ----------------------------------------------------------------- caching


@pytest.mark.parametrize("path", ["/", "/static/app.js", "/static/style.css", "/static/index.html"])
def test_the_page_and_its_files_are_always_revalidated(client, path):
    """
    A browser that reuses an old app.js next to a new index.html shows a button
    with nothing wired to it. no-cache plus the ETag keeps updates honest.
    """
    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers.get("etag")


def test_a_repeat_request_is_a_cheap_304(client):
    etag = client.get("/static/app.js").headers["etag"]

    again = client.get("/static/app.js", headers={"If-None-Match": etag})

    assert again.status_code == 304


def test_the_api_is_not_given_the_page_cache_header(started):
    assert "no-cache" not in started.get("/api/state").headers.get("cache-control", "")
