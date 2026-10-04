# Morning Coding Drill

## Purpose

Build coding fluency by solving short, realistic programming games locally.

The system should train:

1. Python / pandas
2. SQL
3. macOS command line

The goal is not memorizing syntax.

The goal is being able to look at a problem, determine which operations are needed,
combine them correctly, run the code, debug it, and produce the requested result.

---

# Training Model

Each challenge should contain:

## Dataset

Use either:

- synthetic generated data
- curated public data
- Kaggle datasets
- previously generated project data

Synthetic datasets are preferred early because the correct answers can be controlled.

---

## Mission

Give a concrete problem to solve.

Example:

> Among QC-passing samples, identify the three proteins with the largest difference
> in median abundance between AD and control subjects.

Do not initially provide the solution.

---

## Constraints

Challenges may specify constraints such as:

- no loops
- return a DataFrame
- use pandas rather than manual calculation
- do not modify source files
- write output to a specified location

---

# Learning Levels

## Level 1: Guided

Provide:

- relevant functions
- short examples
- direct problem

## Level 2: Selection

Provide function names but do not explain which ones to use.

## Level 3: Retrieval

Provide only:

- dataset
- mission
- constraints

The learner must determine the necessary operations.

## Level 4: Delayed Retrieval

Reuse an older skill several sessions later without announcing that it is being tested.

A concept should not be considered fully mastered until it survives delayed retrieval.

Scheduling this is the progress agent's job: roughly every fourth cleared
challenge it sets level 4 on an older mastered topic and folds the current
topic in as a secondary requirement, without saying so.

---

# Feedback Rules

If the solution is correct:

- confirm briefly
- show a clean reference solution
- point out a more idiomatic solution if relevant

If the solution is incorrect:

- distinguish syntax error from conceptual error
- give one targeted hint
- do not immediately provide the full answer

After two failed attempts:

- provide a stronger hint or partial solution

After three failed attempts:

- explain the solution
- generate a similar challenge immediately

---

# Important Teaching Rules

Require executable code whenever possible.

Do not accept verbal descriptions in place of code.

Do not teach `apply()` when a vectorized pandas operation is preferable.

Accept working but non-idiomatic code, then show the cleaner alternative.

Do not reveal hidden grading logic before the learner submits a solution.

Prefer realistic scientific, biomedical, financial, or analytical examples.

Avoid trivial toy examples involving fruits, colors, or arbitrary letters.

---

# Current Curriculum

## Python

1. DataFrame filtering — mastered
2. GroupBy operations — current
3. Merge / Join
4. Pivot and reshape
5. Missing data and duplicates
6. String operations
7. DateTime operations
8. Apply / map
9. Advanced indexing
10. Performance and vectorization

## SQL

1. SELECT / WHERE — mastered
2. JOIN — current
3. GROUP BY
4. Subqueries
5. CTEs
6. Window functions
7. CASE
8. Set operations
9. Index concepts
10. Complex analytical queries

## Command Line

1. Basic file operations — mastered
2. grep / text processing — current
3. Process management
4. Pipes and redirection
5. find
6. Permissions
7. Environment variables
8. Bash scripting
9. Network utilities
10. Archives

---

# Dataset Philosophy

Do not make datasets large merely for the sake of size.

Prefer datasets with useful analytical complexity:

- missing values
- duplicate records
- inconsistent IDs
- repeated measurements
- multiple related tables
- QC failures
- outliers
- longitudinal measurements
- batch effects

A 100,000-row controlled dataset can be more useful than a multi-gigabyte random dataset.

## A dataset must be able to fail a wrong idea

The first version of this drill asked which disease group had the highest
*median* NEFL intensity. The synthetic AD signal was multiplied by 1.5, which
is strong enough that mean, max and median all return AD — with or without the
QC filter. A learner could hold entirely the wrong idea about medians and still
pass.

So a controlled signal must be tuned to the distinction being taught, not just
made large:

- if the lesson is median vs mean, the column needs enough skew, and the
  groups need to be close enough, that the two disagree
- if the lesson is a filter, dropping the filter must change the answer
- if the lesson is idxmax vs max, the label and the value must both be asked
  for somewhere in the curriculum

`morning_drill/discrimination.py` checks this mechanically: it perturbs a
challenge the way a confused learner would and requires at least one
perturbation to change the answer. A generated challenge that fails the check
is sent back to the trainer agent for revision.

---

# Overall Architecture

Three agents set and review the work. The learner is the human in the loop.

    progress agent  ->  what topic, what level, what to fold back in
    trainer agent   ->  writes the challenge
    the learner     ->  writes the code
    verification    ->  computes the answer, checks the constraints
    grader agent    ->  reviews the code
    progress agent  ->  judges the outcome, briefs the next challenge

The workflow:

    progress agent briefs trainer agent
          ↓
    trainer agent writes mission + spec
          ↓
    spec validated: answerable, not leaking, able to fail a wrong idea
          ↓
    inspect dataset
          ↓
    write solution
          ↓
    verify: answer computed from the spec, constraints checked by ast
          ↓
    grader agent reviews the code
          ↓
    PASS / TRY AGAIN with one escalating hint
          ↓
    progress agent updates progress and briefs the next challenge

## The record

One notebook, re-run and overwritten every morning. It holds no history.

`drill_records.json` holds the history: every challenge, every attempt, the
submitted code, and whatever the learner wrote about the session. The progress
agent reads it before deciding anything, which is what turns the curriculum
from a fixed order into something that responds to evidence:

- days since a topic was last practised drives delayed retrieval
- a repeated error class or root cause means the concept has not landed, so
  the topic repeats and the next challenge aims at that gap
- spec signatures already set are not set again
- the learner's own note outranks the pass rate

Each challenge scores by attempts taken to pass: 1.0, 0.6, 0.3, 0.1, and zero
if never passed. The number is for the learner to watch a trend. The agents
reason from the attempts and the mistakes, not from the score.

## What the agents are not allowed to decide

The trainer agent writes the question but not the grading. It emits a
declarative spec — filters, group_by, value, agg, reduce — and hand-written
code interprets that spec with pandas to get the expected answer, and
generates the reference solution from the same spec. No model-authored code is
ever executed, and the reference the learner is shown cannot drift from what
they were graded against.

The grader agent reviews the code and is told the verdict. It cannot overturn
it. A right answer reached by a confused route is accepted and named as such.

Correctness and compliance are separate. A right answer that used a loop,
used `.apply()` where the mission forbade it, or was typed in as a literal
rather than derived, is not a pass. Those three are decidable, so they are
checked with `ast`, not asked of a model.

Everything deterministic — the verdict, the constraints, the discrimination
check, the structural report on a miss, progress, the log — runs with no API
key. The agents add the question-writing and the prose.

