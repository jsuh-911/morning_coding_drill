"""
Where the drill keeps its state, and how the learner's code is run.

Both are read from the environment so that the same app runs from the repo on
localhost and from a mounted volume in a container, with no code change.
"""

import os
from pathlib import Path

from morning_drill.llm import project_root


def _root():
    return project_root()


def data_dir():
    """
    Directory holding progress.json, drill_records.json and logs/.

    Defaults to the project root, which is where the notebook already writes,
    so running the web app changes nothing about an existing record. Point
    DRILL_DATA_DIR at a mounted volume when hosting: a container filesystem
    resets on deploy, and drill_records.json is the whole adaptive mechanism.
    """
    return Path(os.environ.get("DRILL_DATA_DIR") or _root())


def session_paths():
    """The three path arguments Session takes."""
    base = data_dir()

    return {
        "progress_path": base / "progress.json",
        "records_path": base / "drill_records.json",
        "log_path": base / "logs" / "session.jsonl",
    }


def exec_mode():
    """
    "subprocess" (default) or "inprocess".

    Subprocess gives every run a wall-clock timeout, so an accidental
    `while True` is killed instead of wedging the server. It costs about half
    a second per run (pandas is re-imported) and is the difference between a
    sandbox and an open shell once anyone else can reach the app.

    In-process has no timeout -- a thread running `while True` cannot be
    killed safely -- so it is opt-in, for when run latency matters more.
    """
    mode = (os.environ.get("DRILL_EXEC_MODE") or "subprocess").strip().lower()

    if mode not in {"inprocess", "subprocess"}:
        raise ValueError(f"DRILL_EXEC_MODE must be inprocess or subprocess, got {mode!r}")

    return mode


def exec_timeout():
    return float(os.environ.get("DRILL_EXEC_TIMEOUT") or 15.0)


def exec_memory_mb():
    return int(os.environ.get("DRILL_EXEC_MEMORY_MB") or 2048)


def auth_token():
    """
    Shared secret required on every API call when set. Unset on localhost;
    set it the moment the app is reachable from anywhere else.
    """
    return (os.environ.get("DRILL_AUTH_TOKEN") or "").strip() or None


def agents_enabled():
    return (os.environ.get("DRILL_AGENTS") or "1").strip() not in {"0", "false", "no"}


def app_root():
    """
    The directory the app is importable from -- used as cwd and PYTHONPATH for
    the sandbox child, which runs as `python -m web.runner`.
    """
    return Path(__file__).resolve().parent.parent
