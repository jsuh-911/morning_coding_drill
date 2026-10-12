# Morning Coding Drill

Local coding-practice environment for Python/pandas, SQL and the macOS command
line. Three agents set the work and review it; you write the code.

See [PROJECT.md](PROJECT.md) for the training model and
[ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Optional, to turn the agents on:

```bash
cp .env.example .env
# put ONE provider key in .env
```

`.env` is gitignored. Without a key the drill still runs: challenges come from
a hand-written library, grading and the structural report on a miss are
unchanged, and the prose coaching is what you lose. `session.status()` says
which mode you are in.

## Two front ends

The notebook and the web app run the same `Session`, write the same
`drill_records.json`, and are graded by the same code. Use either.

```bash
./drill            # installs web deps if missing, starts the server, opens the browser
./drill 9000       # different port
```

The question on the left, a sandbox on the right. **Run** executes your code
and costs nothing; **Submit** counts an attempt. See [web/README.md](web/README.md)
for the configuration and for what deploying it beyond localhost involves.

## The daily loop

Open [notebooks/daily_drill.ipynb](notebooks/daily_drill.ipynb) and run it top
to bottom. It is a live sandbox, not a template — overwrite it every morning.
Nothing in the notebook is the record:

```python
from morning_drill import Session

session = Session()
df = session.start()      # progress agent briefs trainer agent

answer = ...              # your pandas pipeline

session.submit(answer)    # grader agent reviews, or the next challenge

session.save(notes="merges still feel unfamiliar")
```

`drill_records.json` is the record, and it is what makes the drill adapt. Every
submission is written as it happens — including your code, since the notebook
gets overwritten — so an interrupted morning loses nothing. `save()` adds the
one thing the score cannot see: your own notes, which the progress agent reads
and weighs against the pass rate.

Tomorrow it starts by reading that file:

- a topic you have not touched in a week comes back at level 4, unannounced
- a mistake you keep repeating aims the next challenge at that gap instead of
  advancing
- a question already set is not set again — spec signatures carry across days
- a note saying something felt shaky counts, even on a clean pass

Each challenge scores by attempts taken: 1.0, 0.6, 0.3, then 0.1; zero if
never passed. The session's `points` is the mean — it is there for you to watch
a trend, while the agents reason from the attempts and the mistakes.

The three agents:

- the **progress agent** picks the topic and level, judges the challenge you
  just finished, and decides when to fold an older skill back in unannounced
- the **trainer agent** writes the challenge
- the **grader agent** reviews your code

Whether the answer is *right* is computed from the challenge's spec, not
judged by a model. A right answer reached with a `for` loop, with `.apply()`
where the mission forbade it, or typed in as a literal, is **not** a pass.

Two checks keep the question and the answer honest:

- **Your code is checked, not only its value.** A correct answer is re-run on
  copies of the data built so that a missing filter, a wrong statistic or a
  wrong cutoff changes the result (`morning_drill/probes.py`). Code that only
  landed on the right label fails, with a hint about which part.
- **The question is written by code from the spec** (`morning_drill/render.py`).
  The trainer agent chooses what to ask (the spec) and may add one sentence of
  setting; the filters with their values, the statistic for each group and
  exactly what to report are then stated the same way every time, so a question
  cannot say "spread" and mean something else, ask for a group while the answer
  is a number, or sit under a title that asks a different question. An
  independent reader rebuilt the spec from the rendered text in 24 of 24 trials.
- **A model can still check a hand-written mission** against its spec
  (`morning_drill/agents/solver.py`): it reads the mission alone and says what
  the answer is and how to compute it.

**Cleaning topics hand you the raw table.** On `missing_duplicates` and
`string_ops` the DataFrame you get is a raw export (`-999` for missing values,
duplicated rows, protein names in mixed case with stray spaces). The expected
answer is computed on the clean table, and every defect is exactly reversible,
so a correct cleaning always lands on it. The rule is the topic, nothing more:
the curriculum order is what decides when you reach it. See
`morning_drill/mess.py`.

The synthetic data is built so that mistakes show: each protein has a different
winning disease group, the mean and the median name different groups in several
cells, and low-QC rows are biased. `python -m scripts.discrimination_yield`
reports how often a random question tells right from wrong.

**Correcting the record.** The Flag button only reaches the question on screen.
A bad question you already moved past keeps counting against you, and the
progress agent keeps reacting to it ("groupby still slipping on..."). Fix it
afterwards:

```bash
./records list                       # recent challenges, with what the grader blamed
./records flag 12 unanswerable       # take question 12 out of the record
./records level groupby 3 active     # put a topic back where it should be
```

Flagging also gives back the attempts it cost you. Restart the web app
afterwards, since it holds progress in memory.

If a question is still wrong, `session.flag("why")` (the **Flag** button in the
web app) takes it out: it scores nothing, any attempts on it are taken back, it
is never dealt again, and a new one replaces it.

The hint escalates each attempt: a question on the first, a line with the key
part blanked on the second, a full walkthrough on the third.

```python
session.reference()          # refuses until you have solved it
session.ask("why idxmax?")   # a question without submitting an attempt
session.flag("why")          # this question is wrong: skip it, count nothing
session.status()             # progress, and this session's history
session.data                 # the current challenge's DataFrame
```

`Session(auto_next=False)` stops it dealing the next challenge on a pass.
`Session(agents=False)` runs on rules alone even with a key present.

Two files, with distinct jobs:

| file | holds |
|---|---|
| `progress.json` | current state: status and level per topic |
| `drill_records.json` | the evidence: every challenge, attempt, and your code |
| `logs/session.jsonl` | one append-only line per attempt, as a crash-safe spine |

[notebooks/agent_session.ipynb](notebooks/agent_session.ipynb) walks through
the machinery; `daily_drill.ipynb` is the one you actually use.

## Datasets outside Python

The drill generates data in memory. For the SQL and command-line work:

```bash
python scripts/generate_dataset.py     # -> datasets/generated/biomarker_study.csv
```

## Tests

```bash
python -m pytest tests/ -q
```

128 tests, no API key needed — the agent paths run against stub models.
