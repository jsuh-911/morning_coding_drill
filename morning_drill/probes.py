"""
Does the code do what the mission said, or did it only land on the right value?

The grader compares a value computed from one dataset. On this data a lot of
wrong pipelines land on the same label: AD wins the highest median NEFL
intensity, and it also wins with the NEFL filter dropped, with the mean, and
with the QC cutoff moved. So a correct answer proves little.

The check here is differential. Rebuild the dataset so that each part of the
spec matters -- rows a filter should remove are made to win if they leak in,
rows at the cutoff are made to win if the boundary is wrong, an outlier is
planted for the statistic -- then run the submitted code on it and compare
with the spec's own answer on the same data. Code that implements the mission
agrees on every probe. Code that only matched on the real data does not.

Nothing here parses the code, so a different but valid route (query(), loc,
sort_values().index[0]) is judged on what it computes.
"""

import numpy as np
import pandas as pd

from .graders import compare_answers
from .spec import FILTER_OPS, evaluate_spec

NUMERIC_CUTOFFS = (">", ">=", "<", "<=")
OUTLIER_AGGS = ("mean", "median", "max", "min", "sum", "std")
LOW_REDUCERS = ("idxmin", "min", "nsmallest")


class _SessionStub:
    """Notebook cells end with session.submit(answer); a probe must not."""

    def submit(self, *args, **kwargs):
        return None


def default_runner(code, frame):
    """
    exec the code on `frame` and return what it bound to `answer`. Raises if
    it fails or binds nothing, which the caller treats as "could not check".

    In-process: right for the notebook, where this code has just run in the
    same kernel. The web app passes its own runner, which adds a timeout.
    """
    ns = {
        "__name__": "__probe__",
        "df": frame,
        "pd": pd,
        "np": np,
        "session": _SessionStub(),
    }
    exec(compile(code, "<submission>", "exec"), ns)

    if "answer" not in ns:
        raise RuntimeError("the code did not bind `answer`")

    return ns["answer"]


# ------------------------------------------------------------------ probes


def _mask(frame, filters):
    keep = pd.Series(True, index=frame.index)

    for f in filters:
        keep &= FILTER_OPS[f.op](frame[f.column], f.value)

    return keep


def _scale(df, spec):
    values = pd.to_numeric(df[spec.value], errors="coerce").abs()

    return float(values.max() or 1.0) * 1000.0


def _loser(df, spec):
    """
    The group that does not win on the real data, and the sign that would make
    it win.
    """
    grouped = evaluate_spec(df, spec.model_copy(update={"reduce": "none"}))
    low = spec.reduce in LOW_REDUCERS

    key = grouped.idxmax() if low else grouped.idxmin()
    sign = -1.0 if low else 1.0

    return (key if isinstance(key, tuple) else (key,)), sign


def _plant(source, spec, key, value, copies, **overrides):
    """
    `copies` rows drawn from `source`, relabelled into group `key` with a
    value that wins, and any column in `overrides` forced.
    """
    rows = source.sample(copies, replace=True, random_state=0).copy()

    for col, k in zip(spec.group_by, key):
        rows[col] = k

    rows[spec.value] = value

    for col, v in overrides.items():
        rows[col] = v

    return rows


def _below(v):
    return v - (abs(v) * 0.01 or 0.01)


def probe_frames(df, spec):
    """
    [(kind, description, frame)] -- each frame built so that one specific
    mistake changes the spec's answer.
    """
    out = []

    try:
        key, sign = _loser(df, spec)
    except Exception:
        return out

    big = sign * _scale(df, spec)
    passing = df[_mask(df, spec.filters)]

    if passing.empty:
        return out

    for i, f in enumerate(spec.filters):
        others = [g for j, g in enumerate(spec.filters) if j != i]
        pool = df[_mask(df, others)]
        excluded = pool[~FILTER_OPS[f.op](pool[f.column], f.value)]

        label = f"{f.column} {f.op} {f.value!r}".replace("None", "").strip()

        if len(excluded):
            rows = _plant(excluded, spec, key, big, len(pool) + 1)
            out.append(("filter", f"the `{label}` condition", pd.concat([df, rows])))

        if f.op in NUMERIC_CUTOFFS and isinstance(f.value, (int, float)):
            n = len(passing) + 1
            on = _plant(passing, spec, key, big, n, **{f.column: f.value})
            out.append(
                ("boundary", f"the edge of `{label}`", pd.concat([df, on]))
            )

            near = _plant(
                passing, spec, key, big, n,
                **{f.column: _below(f.value) if f.op in (">", ">=")
                   else -_below(-f.value)},
            )
            out.append(("cutoff", f"the cutoff in `{label}`", pd.concat([df, near])))

    if spec.agg in OUTLIER_AGGS:
        spike = _plant(passing, spec, key, big * 1000, 1)
        out.append(
            ("statistic", f"the statistic (`{spec.agg}`)", pd.concat([df, spike]))
        )

    return out


# ------------------------------------------------------------------- check


def method_failures(code, df, spec, runner=None):
    """
    The first probe the code disagrees with the spec on, as a student-facing
    sentence -- or [] when the code holds up, or cannot be run to find out.

    One failure is enough and is all that is reported: it is a hint, and the
    hint ladder is meant to be climbed a step at a time.
    """
    if not (code or "").strip():
        return []

    runner = runner or default_runner

    for kind, what, frame in probe_frames(df, spec):
        try:
            truth = evaluate_spec(frame, spec)
            got = runner(code, frame.copy())
        except Exception:
            # Not the learner's fault: a runner limit, a name the probe does
            # not provide. Skipping is safer than failing a correct answer.
            continue

        if not compare_answers(got, truth):
            return [_message(kind, what)]

    return []


def _message(kind, what):
    base = "the value matched, but the code gives a different result when "

    return {
        "filter": base + f"rows that fail {what} are present: it does not apply it",
        "boundary": base + f"rows sit exactly on {what}: check > versus >=",
        "cutoff": base + f"rows sit just outside {what}: check the cutoff",
        "statistic": base + f"one extreme value appears: check {what}",
    }[kind]
