"""
Session: the notebook-facing surface over the three agents.

    session = Session()
    df = session.start()        # progress agent briefs trainer agent
    answer = ...                # the student
    session.submit(answer)      # grader agent reviews, or the next challenge

The student is the human in the loop. Everything else is the graph.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from langgraph.types import Command

from .agents.schemas import GeneratedChallenge
from .progress import (
    ensure_topic,
    load_progress,
    record_attempt,
    save_progress,
    take_back_attempts,
)
from . import records
from .session import BOX_KEY, Box, build_session_graph
from .llm import coach_status, get_model, project_root, text_of


def capture_cell_source():
    """
    Source of the notebook cell that called submit(), so the grader agent can
    review the code and not only the answer.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()

        if ip is None:
            return ""

        history = ip.user_ns.get("In") or []
        return history[-1] if history else ""
    except Exception:
        return ""


class Session:
    def __init__(
        self,
        progress_path=None,
        agents=True,
        auto_next=True,
        log_path=None,
        records_path=None,
    ):
        self.root = project_root()
        self.progress_path = Path(progress_path or self.root / "progress.json")
        self.log_path = Path(log_path or self.root / "logs" / "session.jsonl")
        self.records_path = Path(
            records_path or self.root / records.DEFAULT_FILENAME
        )

        self.progress = load_progress(self.progress_path)
        self.auto_next = auto_next

        model = get_model() if agents else None

        self.models = {
            "progress": model,
            "trainer": model,
            "grader": model,
        }
        self.agents_enabled = model is not None

        self.graph = build_session_graph(
            models=self.models,
            record_fn=self._record,
            apply_fn=self._apply_decision,
        )

        # Everything practised on previous days. This is what makes the
        # progress agent's decision depend on real elapsed time and on
        # mistakes made before today.
        self.digest = records.digest(self.records_path)

        self.box = Box(
            progress=self.progress,
            flagged=self.digest.get("flagged_ids", []),
            avoid=self.digest["avoid"],
            digest=self.digest,
        )

        self.session_id = datetime.now().isoformat(timespec="seconds")
        self.session_record = {
            "session_id": self.session_id,
            "date": self.session_id[:10],
            "started_at": self.session_id,
            "notes": "",
            "challenges": [],
        }
        self._current = None

        self.data = None
        self.challenge = None
        self.thread_id = None
        self._last_state = None

        # the outcome of the previous challenge, fed to the progress agent at
        # the start of the next one. Each challenge is its own graph thread,
        # so this is what carries continuity between them.
        self._outcome = None

    # ------------------------------------------------------------- public

    def start(self):
        """
        Run up to the student: progress agent briefs the trainer agent, the
        trainer agent writes a challenge, the challenge is validated, the
        mission is printed.
        """
        self.thread_id = f"challenge-{uuid.uuid4().hex[:8]}"

        result = self.graph.invoke(
            {"_outcome": self._outcome},
            config=self._config(),
        )

        self._absorb(result)
        self._open_challenge_record()
        self._display_challenge()

        return self.data

    def submit(self, answer=None, code=None):
        """
        Hand an answer to the graph. Returns True on a correct submission.
        """
        if self.thread_id is None:
            raise RuntimeError("No active challenge. Run session.start() first.")

        self.box.answer = answer

        result = self.graph.invoke(
            Command(resume={"code": code if code is not None else capture_cell_source()}),
            config=self._config(),
        )

        self._absorb(result)
        self._display_review()

        passed = bool(self._last_state.get("passed"))

        if passed and self.auto_next:
            print()
            print("-" * 62)
            self.start()

        return passed

    def save(self, notes=""):
        """
        Close out the session: keep your notes, stamp the finish time, and
        write drill_records.json.

        Every submission is already written as it happens, so nothing is lost
        if this is never called -- what this adds is the notes, which the
        progress agent reads and weighs against the pass rate.
        """
        self.session_record["notes"] = notes or self.session_record.get("notes", "")
        self.session_record["finished_at"] = datetime.now().isoformat(
            timespec="seconds"
        )

        records.upsert_session(self.records_path, self.session_record)

        answered = [c for c in self.session_record["challenges"] if c["attempts"]]
        score = records.session_score(answered)

        print()
        print("=" * 62)
        print(f"SESSION {self.session_record['date']}")
        print("=" * 62)
        print()

        for challenge in answered:
            mark = "pass" if challenge["passed"] else "unfinished"
            print(
                f"  {challenge['topic']:<22} L{challenge['level']}  "
                f"{mark:<11} {challenge['attempts']} attempt(s)  "
                f"score {records.challenge_score(challenge['attempts'], challenge['passed'])}"
            )

        print()
        print(
            f"  {score['passed']}/{score['challenges']} passed, "
            f"{score['first_try']} first try, "
            f"mean {score['mean_attempts']} attempts, "
            f"points {score['points']}"
        )

        if notes:
            print()
            print(f"  your note: {notes}")

        print()
        print(f"Written to {self.records_path.name}.")
        print("Tomorrow's progress agent reads it before setting anything.")
        print()

        return score

    def flag(self, reason=""):
        """
        This question is wrong or unanswerable as worded. It counts for
        nothing -- any attempts already spent are taken back out of the
        progress counts and the scores -- it is never dealt again, and the
        next challenge is dealt in its place.
        """
        if self.thread_id is None or self._current is None:
            raise RuntimeError("No active challenge. Run session.start() first.")

        current = self._current
        current["flagged"] = True
        current["flag_reason"] = (reason or "").strip()[:500]
        current["flagged_at"] = datetime.now().isoformat(timespec="seconds")

        self._take_back_attempts(current)

        records.upsert_session(self.records_path, self.session_record)

        challenge_id = current["challenge_id"]

        if challenge_id not in self.box.flagged:
            self.box.flagged.append(challenge_id)

        if current["spec_signature"] not in self.box.avoid:
            self.box.avoid.append(current["spec_signature"])

        self._log_flag(current)

        # The previous challenge's outcome was applied when this one was
        # dealt. Feeding it to the next start() would apply it twice.
        self._outcome = None

        return self.start()

    def _take_back_attempts(self, current):
        if take_back_attempts(
            self.box.progress,
            current.get("domain", "python"),
            current["topic"],
            current.get("attempts_log"),
        ):
            save_progress(self.box.progress, self.progress_path)
            self.progress = self.box.progress

    def _log_flag(self, current):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            with self.log_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "at": current["flagged_at"],
                            "flagged": current["challenge_id"],
                            "reason": current["flag_reason"],
                            "mission": current["mission"],
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

    def ask(self, question):
        model = self.models["grader"]

        if model is None:
            print(f"\n{coach_status()}\n")
            return None

        from .agents import prompts

        prompt = prompts.ASK_PROMPT.format(
            rules=prompts.SHARED_RULES,
            mission=self.challenge["mission"],
            question=question,
        )

        try:
            text = text_of(model.invoke(prompt))
        except Exception as exc:
            print(f"\n(unavailable: {exc})\n")
            return None

        print()
        print(text)
        print()
        return text

    def status(self):
        print()
        print(coach_status())
        print()

        for domain, topics in self.progress.items():
            print(domain.upper())

            for topic, state in topics.items():
                print(
                    f"  {topic:<25} "
                    f"{state.get('status', 'unknown'):<12} "
                    f"Level {state.get('level', '-')}  "
                    f"{state.get('correct', 0)}/{state.get('attempts', 0)}"
                )

            print()

        if self.box.history:
            print("THIS SESSION")

            for item in self.box.history:
                print(
                    f"  {item['topic']} L{item['level']} "
                    f"passed on attempt {item['attempt']}"
                )

            print()

    def reference(self):
        """
        The reference solution for the current challenge. Only after a pass.
        """
        if not (self._last_state or {}).get("passed"):
            print("\nFinish the challenge first.\n")
            return None

        print()
        print(self._last_state["reference_solution"])
        print()
        return self._last_state["reference_solution"]

    # ----------------------------------------------------------- internal

    def _config(self):
        return {
            "configurable": {"thread_id": self.thread_id, BOX_KEY: self.box},
            "recursion_limit": 100,
        }

    def _absorb(self, result):
        self._last_state = result
        self.data = self.box.dataset

        if result.get("challenge"):
            self.challenge = result["challenge"]

    def _record(self, box, entry, state):
        """
        Called by the record node: count the attempt, log it, and remember the
        outcome for the progress agent at the start of the next challenge.
        """
        domain = state["brief"].get("domain", "python")
        entry["domain"] = domain

        ensure_topic(box.progress, domain, entry["topic"])
        record_attempt(box.progress, domain, entry["topic"], entry["passed"])

        save_progress(box.progress, self.progress_path)
        self.progress = box.progress

        self._outcome = entry
        self._update_challenge_record(entry, state)
        self._log(entry, state)

    def _open_challenge_record(self):
        """
        A challenge record exists as soon as one is dealt, but is not written
        until it is attempted -- so an abandoned start() leaves nothing in the
        file.
        """
        challenge = self.challenge

        if not challenge:
            return

        self._current = {
            "challenge_id": GeneratedChallenge(**challenge).id,
            "spec_signature": GeneratedChallenge(**challenge).spec_signature,
            "topic": challenge["topic"],
            "level": challenge["level"],
            "domain": self._last_state.get("brief", {}).get("domain", "python"),
            "mission": challenge["mission"],
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "attempts": 0,
            "passed": False,
            "error_classes": [],
            "root_causes": [],
            "violations": [],
            "hints_used": 0,
            "discrimination": self._last_state.get("discrimination", {}).get(
                "summary"
            ),
            # the notebook is overwritten every morning, so this is the only
            # lasting copy of the work
            "attempts_log": [],
        }

        self.session_record["challenges"].append(self._current)

    def _update_challenge_record(self, entry, state):
        if self._current is None:
            return

        review = state.get("review", {})

        self._current["attempts"] = entry["attempt"]
        self._current["passed"] = entry["passed"]
        self._current["finished_at"] = datetime.now().isoformat(timespec="seconds")

        # Only failures are recorded as mistakes. A passing review carries an
        # error_class of "correct" and a root_cause that describes what went
        # right, and both would pollute the recurring-error counts and the
        # "missed because" lines the progress agent reads.
        if not entry["passed"]:
            if review.get("error_class"):
                self._current["error_classes"].append(review["error_class"])

            if review.get("root_cause"):
                self._current["root_causes"].append(review["root_cause"])

        for violation in state.get("violations", []):
            if violation not in self._current["violations"]:
                self._current["violations"].append(violation)

        if review.get("hint"):
            self._current["hints_used"] += 1

        self._current["attempts_log"].append(
            {
                "attempt": entry["attempt"],
                "passed": entry["passed"],
                "code": state.get("code", ""),
                "error_class": review.get("error_class"),
            }
        )

        records.upsert_session(self.records_path, self.session_record)

    def _apply_decision(self, box, outcome, decision):
        """
        Called by the plan node: persist the progress agent's judgement of the
        challenge that just finished.
        """
        domain = outcome.get("domain", "python")
        topic = outcome["topic"]

        ensure_topic(box.progress, domain, topic)
        topic_state = box.progress[domain][topic]

        if decision.topic_status in ("active", "mastered", "needs_rework"):
            topic_state["status"] = decision.topic_status

        if isinstance(decision.next_level, int):
            topic_state["level"] = decision.next_level

        save_progress(box.progress, self.progress_path)
        self.progress = box.progress

    def _log(self, entry, state):
        """
        One line per attempt. This is the raw material for noticing which
        concepts keep needing hints.
        """
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            record = {
                "at": datetime.now().isoformat(timespec="seconds"),
                **entry,
                "violations": state.get("violations", []),
                "error_class": state.get("review", {}).get("error_class"),
                "discriminating": state.get("discrimination", {}).get("summary"),
            }

            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------ display

    def _display_challenge(self):
        state = self._last_state
        challenge = state.get("challenge")

        if not challenge:
            print("(no challenge was produced)")
            return

        note = state.get("progress_note")

        if note:
            print()
            print(note)

        print()
        print("=" * 62)
        print("MORNING CODING DRILL")
        print("=" * 62)
        print()
        print(
            f"{state['brief'].get('domain', 'python').upper()} "
            f"| {challenge['topic']} "
            f"| Level {challenge['level']}"
        )
        print()
        print(challenge["title"].upper())
        print()
        print("MISSION")
        print()
        print(challenge["mission"])

        if challenge.get("toolbox"):
            print()
            print("TOOLBOX")

            for tool in challenge["toolbox"]:
                print(f"  • {tool}")

        if challenge.get("constraint_ids"):
            from .constraints import CONSTRAINT_CHECKS

            print()
            print("CONSTRAINTS")

            for cid in challenge["constraint_ids"]:
                text = CONSTRAINT_CHECKS.get(cid, (cid, None))[0]
                print(f"  • {text}")

        # Derived from the spec, not from the mission prose: if the two ever
        # disagree, this is the one that is graded.
        from .spec import ChallengeSpec

        try:
            kind = ChallengeSpec(**challenge["spec"]).answer_kind
            print()
            print(f"RETURN  {kind}")
        except Exception:
            pass

        print()
        print("    answer = ...")
        print("    session.submit(answer)")
        print()
        print("=" * 62)
        print()

    def _display_review(self):
        state = self._last_state
        review = state.get("review", {})

        print()

        if state.get("passed"):
            print("PASS")
        else:
            print("TRY AGAIN")
            print(f"attempt {state.get('attempt', 1)}")

        if state.get("correct") and not state.get("passed"):
            print("the answer is right, but the mission's constraints were not met")

        for violation in state.get("violations", []):
            print(f"  constraint: {violation}")

        if not state.get("passed"):
            root = review.get("root_cause")

            if root:
                print()
                print(f"DIAGNOSIS  {review.get('error_class', '?')}")
                print(root)

            hint = review.get("hint")

            if hint and hint != root:
                print()
                print(f"HINT (attempt {state.get('attempt', 1)})")
                print(hint)
        else:
            if review.get("strengths"):
                print()
                print("STRENGTHS")
                print(review["strengths"])

            if review.get("idiomatic_note"):
                print()
                print("NOTE")
                print(review["idiomatic_note"])

            print()
            print("REFERENCE")
            print(state.get("reference_solution", ""))

        if not self.agents_enabled:
            print()
            print(f"({coach_status()})")

        print()
