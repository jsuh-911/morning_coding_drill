"""
drill_records.json -- the durable record of what was practised and how it went.

Two stores, with distinct jobs:

    progress.json        current state: status and level per topic
    drill_records.json   the evidence: every challenge and attempt, with dates

The notebook is a live sandbox that gets overwritten every morning, so this
file is the only lasting copy of the work -- including the submitted code,
which is why each attempt's source is kept here.

Written atomically: a crash mid-write cannot leave a truncated file.
"""

import json
import os
import tempfile
from collections import Counter
from datetime import date, datetime
from pathlib import Path


SCHEMA = 1
DEFAULT_FILENAME = "drill_records.json"

# score by attempts taken to pass; 0 if never passed
ATTEMPT_SCORE = {1: 1.0, 2: 0.6, 3: 0.3}
LATE_SCORE = 0.1

# a mastered topic unpractised for this long is due for delayed retrieval
RETRIEVAL_DUE_DAYS = 7

# A topic can be marked mastered in progress.json without ever appearing in
# the records -- the seeded defaults claim three such. Those claims are not
# evidence, so they only become due for auditing once the drill has this many
# recorded sessions behind it; before that, retrieval waits for a topic we
# have actually watched the learner pass.
UNVERIFIED_AUDIT_AFTER_SESSIONS = 3


def empty():
    return {"schema": SCHEMA, "sessions": []}


def load(path):
    path = Path(path)

    if not path.exists():
        return empty()

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return empty()

    if not isinstance(data, dict) or "sessions" not in data:
        return empty()

    data.setdefault("schema", SCHEMA)
    return data


def save(data, path):
    """
    Atomic write: temp file in the same directory, then os.replace.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")

    try:
        with os.fdopen(handle, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def challenge_score(attempts, passed):
    if not passed:
        return 0.0

    return ATTEMPT_SCORE.get(attempts, LATE_SCORE)


def session_score(challenges):
    """
    Session-level summary. `points` is for the human to watch a trend; the
    progress agent reasons from the raw facts instead.
    """
    answered = [
        c for c in challenges if c.get("attempts") and not c.get("flagged")
    ]

    if not answered:
        return {
            "challenges": 0,
            "passed": 0,
            "first_try": 0,
            "mean_attempts": 0.0,
            "points": 0.0,
        }

    passed = [c for c in answered if c.get("passed")]
    scores = [challenge_score(c["attempts"], c.get("passed")) for c in answered]

    return {
        "challenges": len(answered),
        "passed": len(passed),
        "first_try": len([c for c in passed if c["attempts"] == 1]),
        "mean_attempts": round(
            sum(c["attempts"] for c in answered) / len(answered), 2
        ),
        "points": round(sum(scores) / len(scores), 2),
    }


def upsert_session(path, session):
    """
    Write a session record, replacing any earlier write of the same
    session_id. Challenges that were dealt but never attempted are dropped,
    so an abandoned start() leaves nothing behind.
    """
    session = dict(session)
    # A flagged challenge is kept even with no attempts: the flag is what
    # stops the same question being dealt again tomorrow.
    session["challenges"] = [
        c
        for c in session.get("challenges", [])
        if c.get("attempts") or c.get("flagged")
    ]

    if not session["challenges"]:
        return load(path)

    session["score"] = session_score(session["challenges"])

    data = load(path)
    sessions = [
        s for s in data["sessions"] if s.get("session_id") != session["session_id"]
    ]
    sessions.append(session)
    sessions.sort(key=lambda s: s.get("session_id", ""))

    data["sessions"] = sessions
    save(data, path)

    return data


def challenges(path):
    """
    Every recorded challenge, oldest first, as (session, challenge) pairs.
    """
    return [
        (s, c)
        for s in load(path)["sessions"]
        for c in s.get("challenges", [])
    ]


def flag_in_record(path, challenge_id, reason=""):
    """
    Mark a recorded challenge as wrong. It then counts for nothing in the
    scores or the digest, and is never dealt again. Returns the challenge, or
    None if there is no such id.
    """
    data = load(path)

    for session in data["sessions"]:
        for challenge in session.get("challenges", []):
            if challenge.get("challenge_id") != challenge_id:
                continue

            challenge["flagged"] = True
            challenge["flag_reason"] = (reason or "").strip()[:500]
            challenge["flagged_at"] = datetime.now().isoformat(timespec="seconds")

            session["score"] = session_score(session["challenges"])
            save(data, path)

            return challenge

    return None


def _days_since(stamp, today=None):
    if not stamp:
        return None

    try:
        then = date.fromisoformat(stamp[:10])
    except ValueError:
        return None

    return ((today or date.today()) - then).days


def digest(path, recent=10, avoid=20, today=None):
    """
    A compact summary for the progress agent.

    Deliberately excludes submitted code: it is kept in the file for the
    human and for later analysis, but sending months of source into a prompt
    would crowd out everything else.
    """
    data = load(path)
    sessions = data["sessions"]

    challenges = [
        (s, c)
        for s in sessions
        for c in s.get("challenges", [])
        if c.get("attempts") and not c.get("flagged")
    ]

    # Questions the learner reported as wrong. They count for nothing, in
    # either direction, and are never dealt again.
    flagged = [
        c
        for s in sessions
        for c in s.get("challenges", [])
        if c.get("flagged")
    ]

    topics = {}

    for session, challenge in challenges:
        topic = challenge.get("topic", "unknown")
        state = topics.setdefault(
            topic,
            {
                "challenges": 0,
                "passed": 0,
                "attempts_total": 0,
                "levels_cleared": set(),
                "errors": Counter(),
                "last_practiced": None,
                "recent_misses": [],
            },
        )

        state["challenges"] += 1
        state["attempts_total"] += challenge.get("attempts", 0)

        if challenge.get("passed"):
            state["passed"] += 1
            state["levels_cleared"].add(challenge.get("level"))

        for error in challenge.get("error_classes", []):
            state["errors"][error] += 1

        when = session.get("date") or session.get("session_id", "")[:10]

        if when and (state["last_practiced"] or "") < when:
            state["last_practiced"] = when

        if not challenge.get("passed") or challenge.get("attempts", 1) > 1:
            root = challenge.get("root_causes") or []
            if root:
                state["recent_misses"].append(
                    {"on": when, "why": root[-1], "level": challenge.get("level")}
                )

    for topic, state in topics.items():
        state["pass_rate"] = round(state["passed"] / state["challenges"], 2)
        state["mean_attempts"] = round(
            state["attempts_total"] / state["challenges"], 2
        )
        state["levels_cleared"] = sorted(
            level for level in state["levels_cleared"] if level is not None
        )
        state["recurring_errors"] = dict(state["errors"].most_common(3))
        state["days_since"] = _days_since(state["last_practiced"], today)
        state["recent_misses"] = state["recent_misses"][-3:]

        del state["errors"]
        del state["attempts_total"]

    signatures = [
        c.get("spec_signature")
        for _, c in challenges
        if c.get("spec_signature")
    ]

    recent_rows = [
        {
            "date": s.get("date"),
            "topic": c.get("topic"),
            "level": c.get("level"),
            "passed": c.get("passed"),
            "attempts": c.get("attempts"),
            "score": challenge_score(c.get("attempts", 1), c.get("passed")),
        }
        for s, c in challenges
    ][-recent:]

    return {
        "as_of": (today or date.today()).isoformat(),
        "sessions": len(sessions),
        "topics": topics,
        "avoid": list(
            dict.fromkeys(
                [c["spec_signature"] for c in flagged if c.get("spec_signature")]
                + signatures[-avoid:]
            )
        ),
        "flagged_ids": [c["challenge_id"] for c in flagged if c.get("challenge_id")],
        "recent": recent_rows,
        "notes": [
            {"on": s.get("date"), "note": s["notes"]}
            for s in sessions[-3:]
            if s.get("notes")
        ],
    }


def retrieval_due(digest_data, mastered_topics, exclude=()):
    """
    A mastered topic that has gone unpractised long enough to be worth
    testing again without announcing it. Returns the stalest one, or None.
    """
    if not digest_data:
        return None

    sessions = digest_data.get("sessions", 0)
    candidates = []

    for topic in mastered_topics:
        if topic in exclude:
            continue

        state = digest_data.get("topics", {}).get(topic)
        days = state.get("days_since") if state else None

        if days is None:
            # mastery claimed but never observed here
            if sessions >= UNVERIFIED_AUDIT_AFTER_SESSIONS:
                candidates.append((10**6, topic))
            continue

        if days >= RETRIEVAL_DUE_DAYS:
            candidates.append((days, topic))

    candidates.sort(reverse=True)

    return candidates[0][1] if candidates else None


def digest_text(digest_data):
    """
    Render the digest for the progress agent's prompt.
    """
    if not digest_data or not digest_data["topics"]:
        return "No previous sessions on record. This is the first one."

    lines = [
        f"As of {digest_data['as_of']}, across {digest_data['sessions']} "
        "recorded sessions:",
        "",
    ]

    for topic, state in sorted(digest_data["topics"].items()):
        days = state["days_since"]
        when = (
            "today"
            if days == 0
            else f"{days} days ago"
            if days is not None
            else "unknown"
        )

        lines.append(
            f"  {topic}: {state['passed']}/{state['challenges']} passed, "
            f"mean {state['mean_attempts']} attempts, "
            f"levels cleared {state['levels_cleared'] or 'none'}, "
            f"last practised {when}"
        )

        if state["recurring_errors"]:
            errors = ", ".join(
                f"{k} x{v}" for k, v in state["recurring_errors"].items()
            )
            lines.append(f"      recurring: {errors}")

        for miss in state["recent_misses"]:
            lines.append(f"      missed on {miss['on']}: {miss['why']}")

    if digest_data["notes"]:
        lines.append("")
        lines.append("  what they said about their own sessions:")

        for note in digest_data["notes"]:
            lines.append(f"      {note['on']}: {note['note']}")

    return "\n".join(lines)
