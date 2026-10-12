"""
The morning drill as a web app.

Routes are thin: they run the learner's code, hand the result to Session, and
render whatever Session decided. Every judgement -- the verdict, the
constraints, the hint ladder, the progress decision -- stays in
morning_drill/, unchanged and shared with the notebook.
"""

import threading
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from morning_drill import Session

from . import config, views
from .executor import ANSWER_NAME, run_user_code

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Morning Coding Drill", docs_url=None, redoc_url=None)


@app.middleware("http")
async def revalidate_the_page(request: Request, call_next):
    """
    The page and its script are separate files. With only an ETag and a
    Last-Modified, a browser decides for itself how long to reuse each one, so
    after an update it can pair the new index.html with the old app.js: the
    button is there and nothing is wired to it. no-cache makes it ask every
    time, and the ETag turns that into a cheap 304 when nothing changed.
    """
    response = await call_next(request)

    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"

    return response

# Session holds a LangGraph thread and mutates files on disk; routes run in a
# threadpool, so one lock serialises the whole thing. Single learner, so there
# is nothing to gain from concurrency and a corrupt record to lose.
_lock = threading.Lock()
_session: Session | None = None


# -------------------------------------------------------------------- auth


def require_token(request: Request):
    """
    No-op until DRILL_AUTH_TOKEN is set. Set it before the app is reachable by
    anyone else: an unauthenticated endpoint that execs its POST body is an
    open shell, and separately a way to spend money on the agents.
    """
    expected = config.auth_token()

    if expected is None:
        return

    sent = request.headers.get("x-drill-token") or request.query_params.get("token")

    if not sent or not _constant_time_eq(sent, expected):
        raise HTTPException(status_code=401, detail="Bad or missing token.")


def _constant_time_eq(a, b):
    import hmac

    return hmac.compare_digest(a.encode(), b.encode())


# ----------------------------------------------------------------- session


def _probe_runner(code, frame):
    result = run_user_code(code, frame)

    if not result.submittable:
        raise RuntimeError(result.traceback or "no answer")

    return result.answer


def get_session():
    global _session

    if _session is None:
        _session = Session(
            agents=config.agents_enabled(),
            # The UI deals the next challenge on a button, not automatically,
            # so a pass can be read before the next mission replaces it.
            auto_next=False,
            **config.session_paths(),
        )
        # Probe runs of a submission go through the same sandbox as Run, so
        # they get the same timeout and memory cap.
        _session.box.runner = _probe_runner

    return _session


def _envelope(session, **extra):
    payload = {
        "challenge": views.challenge_view(session),
        "started": session.thread_id is not None,
        "exec_mode": config.exec_mode(),
        "answer_name": ANSWER_NAME,
        "agents_enabled": session.agents_enabled,
    }
    payload.update(extra)
    return payload


# ------------------------------------------------------------------ models


class CodeIn(BaseModel):
    code: str = Field(default="", max_length=200_000)


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class FlagIn(BaseModel):
    reason: str = Field(default="", max_length=500)


class NotesIn(BaseModel):
    notes: str = Field(default="", max_length=10_000)


# ------------------------------------------------------------------ routes


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state", dependencies=[Depends(require_token)])
def state():
    with _lock:
        session = get_session()
        return _envelope(session, review=_last_review(session))


def _last_review(session):
    """
    On a page refresh the browser has nothing; the server still knows whether
    the current challenge has been answered. Only show a review once an
    attempt exists -- a fresh challenge has none.
    """
    if not (session._last_state or {}).get("attempt"):
        return None

    return views.review_view(session)


@app.post("/api/start", dependencies=[Depends(require_token)])
def start():
    """Deal a challenge: the progress agent briefs, the trainer agent writes."""
    with _lock:
        session = get_session()

        try:
            session.start()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Could not set a challenge: {exc}")

        return _envelope(session, review=None)


@app.post("/api/flag", dependencies=[Depends(require_token)])
def flag(body: FlagIn):
    """
    The question is wrong. It counts for nothing, is not dealt again, and the
    next one is dealt in its place.
    """
    with _lock:
        session = get_session()

        if session.thread_id is None:
            raise HTTPException(status_code=409, detail="No challenge yet.")

        try:
            session.flag(body.reason)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Could not flag it: {exc}")

        return _envelope(session, review=None)


@app.post("/api/run", dependencies=[Depends(require_token)])
def run(body: CodeIn):
    """
    Run the code and show what it produced. Costs nothing: no attempt is
    counted, the record is untouched, and the hint ladder does not move. This
    is where the pandas actually gets worked out.
    """
    with _lock:
        session = get_session()

        if session.thread_id is None:
            raise HTTPException(status_code=409, detail="No challenge yet.")

        result = run_user_code(body.code, session.data)

        return {
            "ok": result.ok,
            "stdout": result.stdout,
            "traceback": result.traceback,
            "answer_bound": result.answer_bound,
            "answer": views.preview_of(result.answer) if result.answer_bound else None,
            "value": views.preview_of(result.value) if result.has_value else None,
            "names": result.names,
        }


@app.post("/api/submit", dependencies=[Depends(require_token)])
def submit(body: CodeIn):
    """
    Run the code and hand the answer to the grader. Counts an attempt.

    Code that raises, or that never binds `answer`, is turned away without
    counting one: Session.submit() has nowhere to put a traceback, so a
    crashed run would be recorded as a wrong answer with no explanation. Run
    is there precisely so this is never the interesting case.
    """
    with _lock:
        session = get_session()

        if session.thread_id is None:
            raise HTTPException(status_code=409, detail="No challenge yet.")

        result = run_user_code(body.code, session.data)

        if not result.ok:
            return _envelope(
                session,
                rejected="error",
                reason="The code raised before producing an answer.",
                exec={"stdout": result.stdout, "traceback": result.traceback},
                review=_last_review(session),
            )

        if not result.answer_bound:
            return _envelope(
                session,
                rejected="no_answer",
                reason=(
                    f"Nothing called `{ANSWER_NAME}` was bound. "
                    f"Assign the result to `{ANSWER_NAME}`."
                ),
                exec={"stdout": result.stdout, "traceback": None},
                review=_last_review(session),
            )

        try:
            session.submit(result.answer, code=body.code)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Grading failed: {exc}")

        return _envelope(
            session,
            rejected=None,
            exec={"stdout": result.stdout, "traceback": None},
            review=views.review_view(session),
            answer=views.preview_of(result.answer),
        )


@app.post("/api/ask", dependencies=[Depends(require_token)])
def ask(body: AskIn):
    """A question, without spending an attempt."""
    with _lock:
        session = get_session()

        if session.thread_id is None:
            raise HTTPException(status_code=409, detail="No challenge yet.")

        text = session.ask(body.question)

        return {"text": text, "agents_enabled": session.agents_enabled}


@app.get("/api/reference", dependencies=[Depends(require_token)])
def reference():
    with _lock:
        session = get_session()
        text = session.reference()

        if text is None:
            raise HTTPException(status_code=409, detail="Finish the challenge first.")

        return {"reference": text}


@app.get("/api/status", dependencies=[Depends(require_token)])
def status():
    with _lock:
        session = get_session()
        payload = views.status_view(session)
        payload["exec_mode"] = config.exec_mode()
        payload["data_dir"] = str(config.data_dir())
        return payload


@app.post("/api/save", dependencies=[Depends(require_token)])
def save(body: NotesIn):
    """
    Write the session notes. The one thing the score cannot see, and the
    progress agent weighs it against the pass rate tomorrow.
    """
    with _lock:
        session = get_session()
        score = session.save(notes=body.notes)
        return {"score": score, "records_path": str(session.records_path)}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
