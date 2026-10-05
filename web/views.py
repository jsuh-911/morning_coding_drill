"""
Session state -> JSON for the browser.

This is the one place that reads Session._last_state. The notebook renders the
same information by printing it (Session._display_challenge and
_display_review); this renders it as data instead. Nothing under
morning_drill/ is modified, so the notebook and the web app stay in step.
"""

import math

import pandas as pd

from morning_drill.agents.schemas import GeneratedChallenge
from morning_drill.constraints import CONSTRAINT_CHECKS
from morning_drill.llm import coach_status
from morning_drill.records import challenge_score, session_score
from morning_drill.spec import ChallengeSpec

PREVIEW_ROWS = 12


# ------------------------------------------------------------------ tables


def _cell(value):
    """A JSON-safe cell. NaN and numpy scalars do not survive json.dumps."""
    if value is None:
        return None

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None

    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)

    item = getattr(value, "item", None)

    if callable(item):
        try:
            return _cell(item())
        except Exception:
            return str(value)

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def table_of(obj, limit=PREVIEW_ROWS):
    """
    A DataFrame, Series or scalar as {columns, rows, shape, truncated}, so the
    browser can lay it out instead of receiving pre-rendered text.
    """
    if isinstance(obj, pd.Series):
        frame = obj.rename("value").to_frame()
        index_name = obj.index.name or ""
    elif isinstance(obj, pd.DataFrame):
        frame = obj
        index_name = obj.index.name or ""
    else:
        return None

    head = frame.head(limit)
    show_index = not isinstance(frame.index, pd.RangeIndex)

    columns = ([index_name or "index"] if show_index else []) + [
        str(c) for c in head.columns
    ]

    rows = []

    for key, row in head.iterrows():
        cells = [_cell(key)] if show_index else []
        cells += [_cell(v) for v in row.tolist()]
        rows.append(cells)

    return {
        "columns": columns,
        "rows": rows,
        "shape": list(frame.shape),
        "truncated": len(frame) > limit,
        "dtypes": {str(c): str(t) for c, t in head.dtypes.items()},
    }


def preview_of(obj):
    """What Run shows for whatever the code put in `answer`."""
    if obj is None:
        return {"kind": "none", "repr": "None", "table": None}

    kind = type(obj).__name__
    table = table_of(obj)

    try:
        text = repr(obj)
    except Exception as exc:
        text = f"(unrepresentable: {exc})"

    return {
        "kind": kind,
        "repr": text[:4000],
        "table": table,
    }


def dataset_view(df):
    if df is None:
        return None

    view = table_of(df)

    if view is None:
        return None

    view["describe"] = _describe(df)
    return view


def _describe(df):
    """Numeric summary, for spotting the skew and the outliers on sight."""
    try:
        numeric = df.select_dtypes("number")

        if numeric.empty:
            return None

        return table_of(numeric.describe().round(3), limit=12)
    except Exception:
        return None


# ------------------------------------------------------- challenge / review


def challenge_view(session):
    state = session._last_state or {}
    challenge = state.get("challenge")

    if not challenge:
        return None

    constraints = [
        CONSTRAINT_CHECKS.get(cid, (cid, None))[0]
        for cid in challenge.get("constraint_ids", [])
    ]

    try:
        return_kind = ChallengeSpec(**challenge["spec"]).answer_kind
    except Exception:
        return_kind = None

    # `id` is a property on GeneratedChallenge, so it is absent from the
    # dumped dict -- the same reason answer_kind has to be recomputed here.
    try:
        challenge_id = GeneratedChallenge(**challenge).id
    except Exception:
        challenge_id = None

    return {
        "id": challenge_id,
        # What the browser keys the editor on: one dealt challenge, so a miss
        # keeps your code and a new deal starts clean. The id alone is a hash
        # of the question, which a re-deal of the same question would share.
        "key": f"{session.thread_id}:{challenge_id}",
        "domain": (state.get("brief") or {}).get("domain", "python"),
        "topic": challenge.get("topic"),
        "level": challenge.get("level"),
        "title": challenge.get("title"),
        "mission": challenge.get("mission"),
        "toolbox": challenge.get("toolbox") or [],
        "constraints": constraints,
        "return_kind": return_kind,
        "progress_note": state.get("progress_note"),
        "attempt": state.get("attempt", 0),
        "dataset": dataset_view(session.data),
    }


def review_view(session):
    """
    Mirrors Session._display_review. `reference` is only ever included on a
    pass -- the notebook's reference() refuses before that, and so does this.
    """
    state = session._last_state or {}
    review = state.get("review") or {}
    passed = bool(state.get("passed"))

    out = {
        "passed": passed,
        "correct": bool(state.get("correct")),
        "attempt": state.get("attempt", 1),
        "violations": list(state.get("violations") or []),
        "answer_repr": state.get("answer_repr"),
        "agents_enabled": session.agents_enabled,
        "coach_status": coach_status(),
    }

    if passed:
        out.update(
            {
                "strengths": review.get("strengths"),
                "idiomatic_note": review.get("idiomatic_note"),
                "reference": state.get("reference_solution"),
            }
        )
    else:
        out.update(
            {
                "error_class": review.get("error_class"),
                "root_cause": review.get("root_cause"),
                "hint": review.get("hint"),
                # Right answer, forbidden route: the mission's own rules, not
                # a grading quirk, so it is said separately from a wrong answer.
                "constraints_only": bool(state.get("correct")) and not passed,
            }
        )

    return out


# ------------------------------------------------------------------ status


def status_view(session):
    domains = []

    for domain, topics in (session.progress or {}).items():
        domains.append(
            {
                "domain": domain,
                "topics": [
                    {
                        "topic": topic,
                        "status": meta.get("status", "unknown"),
                        "level": meta.get("level"),
                        "correct": meta.get("correct", 0),
                        "attempts": meta.get("attempts", 0),
                    }
                    for topic, meta in topics.items()
                ],
            }
        )

    answered = [c for c in session.session_record["challenges"] if c["attempts"]]

    return {
        "coach_status": coach_status(),
        "agents_enabled": session.agents_enabled,
        "exec_mode": None,  # filled in by the app, which owns config
        "domains": domains,
        "today": {
            "date": session.session_record["date"],
            "notes": session.session_record.get("notes", ""),
            "score": session_score(answered) if answered else None,
            "challenges": [
                {
                    "topic": c["topic"],
                    "level": c["level"],
                    "passed": c["passed"],
                    "attempts": c["attempts"],
                    "score": challenge_score(c["attempts"], c["passed"]),
                }
                for c in answered
            ],
        },
    }
