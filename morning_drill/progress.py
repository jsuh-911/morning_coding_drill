import copy
import json
from datetime import datetime
from pathlib import Path


DEFAULT_PROGRESS = {
    "python": {
        "dataframe_filtering": {
            "status": "mastered",
            "level": 4,
        },
        "groupby": {
            "status": "active",
            "level": 1,
            "attempts": 0,
            "correct": 0,
        },
    },
    "sql": {
        "select_where": {
            "status": "mastered",
            "level": 4,
        },
        "joins": {
            "status": "active",
            "level": 1,
            "attempts": 0,
            "correct": 0,
        },
    },
    "cli": {
        "file_operations": {
            "status": "mastered",
            "level": 4,
        },
        "text_processing": {
            "status": "active",
            "level": 1,
            "attempts": 0,
            "correct": 0,
        },
    },
}


def load_progress(path="progress.json"):
    """
    Read progress.json. Create a default file if it does not exist.
    """
    path = Path(path)

    if not path.exists():
        progress = copy.deepcopy(DEFAULT_PROGRESS)
        save_progress(progress, path)
        return progress

    with path.open("r") as f:
        return json.load(f)


def save_progress(progress, path="progress.json"):
    """
    Save progress dictionary to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(progress, f, indent=2)


def record_attempt(progress, domain, topic, success):
    """
    Update attempts and successes for a topic.
    """
    skill = progress[domain][topic]
    skill["attempts"] = skill.get("attempts", 0) + 1

    if success:
        skill["correct"] = skill.get("correct", 0) + 1

    skill["last_practiced"] = datetime.now().isoformat(timespec="seconds")
    return progress


def ensure_topic(progress, domain, topic):
    """
    A generated challenge may name a curriculum topic with no entry yet.
    """
    progress.setdefault(domain, {})
    progress[domain].setdefault(
        topic,
        {"status": "active", "level": 1, "attempts": 0, "correct": 0},
    )
    return progress


def take_back_attempts(progress, domain, topic, attempts_log):
    """
    Undo what record_attempt counted for a challenge that turned out to be
    wrong. Returns True if there was anything to undo.
    """
    skill = progress.get(domain, {}).get(topic)

    if not skill or not attempts_log:
        return False

    skill["attempts"] = max(0, skill.get("attempts", 0) - len(attempts_log))
    skill["correct"] = max(
        0, skill.get("correct", 0) - sum(1 for a in attempts_log if a.get("passed"))
    )

    return True
