# Three agents, one student

Current design, verified against a real model (`claude-sonnet-5`) on
2026-10-04. The earlier single-coach `Drill` has been removed; this is the
only system.

## The shape of it

    prepare -> plan(progress agent) -> author(trainer agent) -> present
                                                                   |
                                     +-----------------------------+
                                     v
                              await_submission   <--- the student, interrupt()
                                     |
                                  verify     deterministic
                                     |
                                  grade      grader agent
                                     |
                                  record
                                     |
                          passed? --+-- no --> back to await_submission
                                     |
                                    yes --> END

One graph run is one challenge, every retry included. The retry loop is a real
loop in the graph now, because `interrupt()` is what waits for the student —
that is the whole difference from the previous design, where each attempt was
a separate invocation and the loop closed through notebook cells.

Choosing the *next* challenge is the next run, with a fresh thread. The
outcome of the finished challenge is handed forward by `Session` so the
progress agent can judge it.

## Who decides what

| | decides | cannot decide |
|---|---|---|
| **progress agent** | what topic and level comes next, status of the topic just finished, when to fold an older skill back in | whether an answer was right |
| **trainer agent** | the mission, the toolbox, the constraints, the spec | how the answer is checked |
| **grader agent** | constraint reading, soundness of the reasoning, idiom, strengths, the hint | whether the answer was right |
| **`morning_drill.spec`** | the correct answer | anything pedagogical |
| **the student** | the code | — |

### Why the grader agent cannot mark its own homework

An LLM that invents the question and also judges the answer has no ground
truth, so PASS becomes a matter of mood and `progress.json` stops meaning
anything.

So the trainer agent emits a **spec** next to its prose:

    filters -> groupby(group_by)[value].agg(agg) -> reduce

`morning_drill/spec.py` — hand-written, no model involved — interprets that
with pandas to get the expected answer, and generates the reference solution
from the same spec, so the code the student is shown cannot drift from what
they were graded against. A test runs the generated reference code and asserts
it reproduces the spec's answer.

Specs are validated against the dataset's declared schema before anything
runs: `group_by` must be categorical, `value` numeric, filter values must
exist in the column's category list. A malformed or adversarial spec is
rejected, not executed. **No model-authored code is ever executed.**

The grader agent is told the verdict and cannot overturn it. It judges the
code: a right answer reached by a confused route is `style_only`, and it has
to say why.

## Correct is not the same as passed

`verify` returns both. `correct` is about the value; `passed` also requires
the constraints the mission stated. A right answer reached with a `for` loop
is correct and not a pass — and `.apply()` under a `no_apply` constraint, and
an answer typed in as a literal rather than derived.

Those three checks are decidable, so they are done with `ast` in
`morning_drill/constraints.py`, not asked of a model. `looks_hardcoded` fires
only when the expected answer appears as a literal *and* no pandas work was
done, so a legitimate solution mentioning a threshold is not accused.

## Challenges have to be able to fail you

The original drill asked for a median, but the synthetic AD/NEFL signal was so
strong that `mean`, `max` and `median` all returned `AD`. A student could hold
the wrong idea and pass.

So `morning_drill/discrimination.py` perturbs a spec the way a confused
student would — swap the statistic, drop a filter, flip `idxmax` to `idxmin` —
and requires that at least one perturbation changes the answer. The trainer
agent's output is rejected and sent back for revision if it is not
discriminating, with the reason stated so it can fix it. The first attempt is
held to the stricter bar of catching *every* perturbation.

This also graded the hand-written library: three of four entries were tightened
on the strength of it, and the fourth (`Highest NEFL group`) stays weak because
its weakness is in the generator, not the spec.

## The record is what makes it adapt

A session is one morning in a notebook that gets overwritten. Continuity lives
in `drill_records.json`:

    progress.json         current state: status and level per topic
    drill_records.json    the evidence: every challenge, every attempt, the code
    logs/session.jsonl    append-only spine, one line per attempt

Written atomically (temp file, then `os.replace`) and upserted on **every
submission**, not only at the end, so a kernel crash costs nothing.
`session.save(notes=...)` adds the learner's own account of the session, which
is the one input no score can derive.

`records.digest()` renders it for the progress agent: per topic, pass rate,
mean attempts, levels cleared, **days since last practised**, recurring error
classes, and the last few root causes. Plus the spec signatures of recent
challenges, which become `Brief.avoid` — this is what stops the trainer agent
re-asking yesterday's question, a thing it did constantly when `avoid` only
spanned one session.

The digest deliberately **excludes the submitted code**. The code is kept in
the file, because the notebook is overwritten and it is the learner's only
lasting copy, but months of source in a prompt would crowd out everything
else.

Scoring is by attempts taken to pass — 1.0, 0.6, 0.3, 0.1, and 0 if never
passed. A constraint violation needs no separate penalty: it already costs an
attempt, since a violation cannot pass.

### Two traps in the records layer

**`hash()` is salted per interpreter.** `GeneratedChallenge.id` used it, so the
same mission produced a different id on every run and nothing recorded
yesterday ever matched anything generated today. It is `hashlib.sha256` now,
and a test asserts stability across two `PYTHONHASHSEED` values.

**Claimed mastery is not observed mastery.** `progress.json` ships with three
topics marked mastered that this drill has never watched anyone pass. Treating
them as infinitely stale made the very first cleared challenge jump to level 4.
A topic with no record is only audited once there are
`UNVERIFIED_AUDIT_AFTER_SESSIONS` sessions to audit against; before that,
retrieval waits for a topic we have actually seen passed.

## Delayed retrieval has an owner

PROJECT.md's level 4 — reuse an old skill without announcing it — is a
scheduling decision, so it belongs to the progress agent. With records to go
on it picks the stalest mastered topic unpractised for a week or more and puts
the current topic in `fold_in` instead; with no records yet it falls back to
every fourth cleared challenge. A topic is mastered only once it has
survived level 4; passing at level 1 is not mastery. Three or more attempts
means `needs_rework`: same topic, same level.

## Everything runs without a key

`get_model()` returning `None` is a supported state throughout:

- progress agent -> `fallback_decision`, the same policy as rules
- trainer agent -> the hand-written library in `agents/library.py`
- grader agent -> a deterministic review built from the violations and the
  structural report

The deterministic parts — the verdict, the constraints, the discrimination
check, the structural report on a miss, progress, the log — are the same
either way. The agents add the question-writing and the prose.

## Unserializable objects

The checkpointer puts state *and* plain `config["configurable"]` values
through msgpack. A DataFrame raises there, and so does a bare `numpy.float64`,
which is what `.median().max()` returns. So the dataset and the student's
answer live on a `Box` in config, which nodes read and write directly and
which is never serialized. Nothing about them is worth replaying anyway.

## Files

    morning_drill/
      spec.py                  ChallengeSpec, evaluate_spec, reference codegen
      constraints.py           ast checks: loops, apply, typed-in answers
      discrimination.py        is this challenge able to fail a wrong idea
      curriculum.py            PROJECT.md's curriculum as data
      graders.py               compare_answers, discrepancy_report
      generators.py            the synthetic biomarker dataset
      progress.py              progress.json read/write
      llm.py                   .env loading, get_model(), text_of()
      records.py               drill_records.json, scoring, digest()
      session.py               the graph, Box, nodes
      drill_session.py         Session: the notebook surface
      agents/
        schemas.py             Brief, GeneratedChallenge, Review, ProgressDecision
        prompts.py             the three prompts + the hint-strength table
        progress_agent.py      decide()  + rule-based fallback
        trainer_agent.py       author()  + validate() + revision loop
        grader_agent.py        verify()  (deterministic) + review()
        library.py             hand-written challenges, used with no key

    notebooks/daily_drill.ipynb    the live sandbox, run daily
    notebooks/agent_session.ipynb  a walkthrough of the machinery
    scripts/generate_dataset.py    the CSV, for the SQL and CLI drills
    tests/                         109 tests, no API key required

## What the first real run changed

Four things only a live model exposed.

**`temperature` is rejected outright.** Current Claude models (Sonnet 5, the
Opus 4.6+ family) return a 400 on it. The trainer agent's first call failed,
the library fallback caught it, and the drill carried on — which is the
fallback working, but it would have run on library challenges forever with
only a line in `trainer_notes` to say why. `llm.py` no longer sends
`temperature`; set `TRAINER_TEMPERATURE` only for an older model.

**`.content` is not a string.** Thinking is always on, so a response is a list
of content blocks with the thinking text empty under the default display
setting. `session.ask()` was printing the raw block list, signature and all.
`text_of()` in `llm.py` extracts the text blocks.

**The grader agent quoted internal ids at the student** — "Violates no_loops
outright". It now receives the same wording the student was shown.

**The mission and the spec can disagree.** The trainer agent wrote "report both
the winning disease and its median intensity value for each protein" against a
spec whose `idxmax` returns the single tuple `('NEFL', 'AD')`. A student
following the prose would have failed with a defensible answer. Three changes:
the prompt now states what each `reduce` actually returns, `validate` rejects a
mission asking for two outputs when one is returned, and the challenge display
prints `RETURN <shape>` **derived from the spec** — so if the two ever disagree
again, the student still knows what is graded.

The revision loop itself worked on the first real attempt: the agent's opening
challenge was rejected for catching only two of four plausible mistakes, and
its revision was accepted.

## What is not done

- **Prompt quality has had a couple of sessions' observation, not tuning.**
  The escalation reads well (a question, then a blanked line, then a full
  walkthrough) and the strengths are specific, but n is small.
- **The cross-day loop has been simulated, not lived.** Two consecutive
  sessions were run end to end — the second read the first's record and its
  note, said so, and set a different question — but no real multi-week history
  exists yet. The level-4-after-a-week rule in particular has never fired on
  genuinely elapsed time.
- The three agents share one model. Per-agent selection — something cheap for
  progress, something stronger for the trainer — is a `models` dict away in
  `Session.__init__`.
- Only `python/groupby` and `python/dataframe_filtering` have a spec
  vocabulary. SQL and CLI need their own generators and spec types; the
  curriculum marks the rest `(no generator yet)` and the trainer agent is
  confined to what exists.
- `Highest NEFL group`, the one library challenge kept from the original
  drill, still cannot distinguish median from mean. The fix is in the
  generator — weaken the 1.5x multiplier, add outliers to another group — and
  it changes the challenge currently in progress, so it is left as a decision.
  The discrimination check reports it as catching 1 of 4 mistakes.
