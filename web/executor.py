"""
Running the learner's code.

Everything the rest of the app knows about execution is run_user_code(). Swap
what happens inside it -- in-process, subprocess, a container, a remote
sandbox -- and nothing else changes.
"""

import ast
import io
import pickle
import subprocess
import sys
import traceback as tb
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Optional

from . import config

ANSWER_NAME = "answer"

STDOUT_LIMIT = 20_000


@dataclass
class ExecResult:
    """
    ok              the code ran without raising
    answer_bound    the code bound a variable called `answer`
    answer          that value, ready to hand to Session.submit()
    """

    ok: bool
    stdout: str = ""
    traceback: Optional[str] = None
    answer: Any = None
    answer_bound: bool = False
    value: Any = None          # a trailing bare expression, echoed back
    has_value: bool = False
    names: list = field(default_factory=list)

    @property
    def submittable(self):
        return self.ok and self.answer_bound


def _namespace(df):
    import numpy as np
    import pandas as pd

    return {
        "__name__": "__drill__",
        "df": df,
        "pd": pd,
        "np": np,
    }


def _public_names(ns):
    seeded = {"__name__", "__builtins__", "df", "pd", "np"}
    return sorted(k for k in ns if k not in seeded and not k.startswith("_"))


def _split_trailing_expression(code):
    """
    Everything but a trailing bare expression, and that expression on its own.

    `df.head()` on the last line is what anyone types first in a sandbox, and
    in a notebook it shows the frame. Under plain exec() it evaluates and the
    value is thrown away, which looks like the tool ignoring you.
    """
    tree = ast.parse(code, filename="<submission>", mode="exec")

    # Always compile, never hand the raw string back: exec() on a string
    # names the frames "<string>", and _clean_traceback keeps only
    # "<submission>" frames -- so the failing line would vanish.
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return compile(tree, "<submission>", "exec"), None

    last = tree.body[-1]
    head = ast.Module(body=tree.body[:-1], type_ignores=tree.type_ignores)

    return (
        compile(ast.fix_missing_locations(head), "<submission>", "exec"),
        compile(ast.Expression(last.value), "<submission>", "eval"),
    )


def run_inprocess(code, df):
    """
    exec in this process. Right for localhost and nothing else: there is no
    timeout here, because a thread running `while True` cannot be killed
    safely -- an accidental infinite loop wedges the server until Ctrl-C.
    """
    ns = _namespace(df)
    out = io.StringIO()
    value = None
    has_value = False

    try:
        body, tail = _split_trailing_expression(code)

        with redirect_stdout(out), redirect_stderr(out):
            exec(body, ns)

            if tail is not None:
                value = eval(tail, ns)
                has_value = value is not None
    except BaseException:
        return ExecResult(
            ok=False,
            stdout=out.getvalue()[:STDOUT_LIMIT],
            traceback=_clean_traceback(),
            names=_public_names(ns),
        )

    return ExecResult(
        ok=True,
        stdout=out.getvalue()[:STDOUT_LIMIT],
        answer=ns.get(ANSWER_NAME),
        answer_bound=ANSWER_NAME in ns,
        value=value,
        has_value=has_value,
        names=_public_names(ns),
    )


def run_subprocess(df, code):
    """
    exec in a short-lived child with a CPU, memory and wall-clock cap. Use
    this whenever anyone but the person at the keyboard can reach the app.

    This caps resources; it does not cut off the network or the filesystem.
    For that, run the child in a container with --network=none and a
    read-only mount, which is this same protocol over a different transport.
    """
    payload = pickle.dumps({"code": code, "df": df}, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "web.runner"],
            input=payload,
            capture_output=True,
            timeout=config.exec_timeout(),
            cwd=str(config.app_root()),
            env={
                "DRILL_EXEC_MEMORY_MB": str(config.exec_memory_mb()),
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(config.app_root()),
            },
        )
    except subprocess.TimeoutExpired:
        return ExecResult(
            ok=False,
            traceback=(
                f"Timed out after {config.exec_timeout():.0f}s. "
                "An unbounded loop, or a step heavier than it needs to be."
            ),
        )

    if proc.returncode != 0 or not proc.stdout:
        detail = (proc.stderr or b"").decode("utf-8", "replace")[-2000:]

        return ExecResult(
            ok=False,
            traceback=(
                "The sandbox exited before returning a result "
                f"(code {proc.returncode}).\n{detail}"
            ).strip(),
        )

    try:
        result = pickle.loads(proc.stdout)
    except Exception as exc:
        return ExecResult(ok=False, traceback=f"Could not read the sandbox result: {exc}")

    return ExecResult(
        ok=result["ok"],
        stdout=result.get("stdout", "")[:STDOUT_LIMIT],
        traceback=result.get("traceback"),
        answer=result.get("answer"),
        answer_bound=result.get("answer_bound", False),
        value=result.get("value"),
        has_value=result.get("has_value", False),
        names=result.get("names", []),
    )


def run_user_code(code, df):
    """The one entry point. Everything above is an implementation detail."""
    if not (code or "").strip():
        return ExecResult(ok=False, traceback="Nothing to run.")

    if config.exec_mode() == "subprocess":
        return run_subprocess(df, code)

    return run_inprocess(code, df)


def _clean_traceback():
    """
    The learner's traceback, without the web app's own frames above it.
    """
    exc_type, exc, exc_tb = sys.exc_info()

    # A SyntaxError is raised by compile(), so there are no <submission>
    # frames to keep -- every frame belongs to this file. Showing the
    # exception alone is both shorter and the whole of what went wrong.
    if issubclass(exc_type, SyntaxError):
        return "".join(tb.format_exception_only(exc_type, exc)).rstrip()

    frames = [f for f in tb.extract_tb(exc_tb) if f.filename == "<submission>"]

    lines = ["Traceback (most recent call last):\n"]
    lines += tb.format_list(frames)
    lines += tb.format_exception_only(exc_type, exc)

    return "".join(lines).rstrip()
