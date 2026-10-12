"""
Look at, and correct, the record.

A failed attempt counts against you wherever it came from, including attempts
on a question that was wrong. The Flag button only reaches the question on
screen, so a bad question you already moved past keeps feeding the progress
agent ("groupby still slipping on...") until it is flagged here.

    python -m scripts.records list [--all] [-n 15]
    python -m scripts.records flag <number|id> [why it was wrong]
    python -m scripts.records level <topic> <1-4> [active|mastered|needs_rework]

Stop the web app first, or restart it straight afterwards: it keeps progress in
memory and would write its own copy back over these edits.
"""

import argparse
import os
import sys

from morning_drill import records
from morning_drill.progress import (
    ensure_topic,
    load_progress,
    save_progress,
    take_back_attempts,
)
from web.config import session_paths

STATUSES = ("active", "mastered", "needs_rework")


def _paths():
    paths = session_paths()
    return paths["records_path"], paths["progress_path"]


def _rows(records_path):
    return [
        {"n": i, "session": s, "c": c}
        for i, (s, c) in enumerate(records.challenges(records_path), start=1)
    ]


def _outcome(c):
    if c.get("flagged"):
        return "FLAGGED"

    if not c.get("attempts"):
        return "-"

    return "pass" if c.get("passed") else "FAIL"


def cmd_list(args):
    records_path, _ = _paths()
    rows = _rows(records_path)

    if not args.all:
        rows = [r for r in rows if r["c"].get("attempts") or r["c"].get("flagged")]

    for r in rows[-args.n:]:
        c = r["c"]
        print(
            f"{r['n']:>3}  {r['session'].get('date', '?')}  "
            f"{c.get('domain', 'python')}/{c.get('topic', '?')} L{c.get('level', '?')}  "
            f"{c.get('attempts', 0)} att  {_outcome(c):<7}  {c.get('challenge_id', '')}"
        )
        print(f"       {(c.get('mission') or '')[:110]}")

        causes = c.get("root_causes") or []

        if causes and not c.get("passed") and not c.get("flagged"):
            print(f"       grader said: {causes[-1][:110]}")

    print("\nflag a wrong question with:  python -m scripts.records flag <number> <why>")
    return 0


def cmd_flag(args):
    records_path, progress_path = _paths()
    rows = _rows(records_path)

    target = None

    if args.which.isdigit():
        target = next((r["c"] for r in rows if r["n"] == int(args.which)), None)
    else:
        target = next((r["c"] for r in rows if r["c"].get("challenge_id") == args.which), None)

    if target is None:
        print(f"no challenge {args.which!r}; run `list` first", file=sys.stderr)
        return 1

    if target.get("flagged"):
        print("already flagged")
        return 0

    reason = " ".join(args.reason) or "flagged from the command line"
    records.flag_in_record(records_path, target["challenge_id"], reason)

    progress = load_progress(progress_path)
    undone = take_back_attempts(
        progress,
        target.get("domain", "python"),
        target["topic"],
        target.get("attempts_log"),
    )

    if undone:
        save_progress(progress, progress_path)

    print(
        f"flagged {target['challenge_id']}: {target.get('mission', '')[:90]}\n"
        f"  {len(target.get('attempts_log') or [])} attempt(s) taken back from "
        f"{target.get('domain', 'python')}/{target['topic']}"
        if undone
        else f"flagged {target['challenge_id']} (no progress counts to undo)"
    )
    print("restart the web app so it reloads the record.")
    return 0


def cmd_level(args):
    _, progress_path = _paths()

    if args.status and args.status not in STATUSES:
        print(f"status must be one of {STATUSES}", file=sys.stderr)
        return 1

    progress = load_progress(progress_path)
    ensure_topic(progress, "python", args.topic)
    state = progress["python"][args.topic]
    before = (state.get("status"), state.get("level"))

    state["level"] = args.level

    if args.status:
        state["status"] = args.status

    save_progress(progress, progress_path)

    print(f"python/{args.topic}: {before[0]} level {before[1]} -> {state['status']} level {state['level']}")
    print("restart the web app so it reloads the progress.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="records", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="recent challenges and how they went")
    p.add_argument("--all", action="store_true", help="include challenges never attempted")
    p.add_argument("-n", type=int, default=15)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("flag", help="mark a past challenge as wrong")
    p.add_argument("which", help="the number from `list`, or the challenge id")
    p.add_argument("reason", nargs="*")
    p.set_defaults(fn=cmd_flag)

    p = sub.add_parser("level", help="set a topic's level (and status)")
    p.add_argument("topic")
    p.add_argument("level", type=int, choices=[1, 2, 3, 4])
    p.add_argument("status", nargs="?")
    p.set_defaults(fn=cmd_level)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
