"""
The question, written by code from the spec.

A mission the model writes freely can say "spread", "gap" or "widest" and mean
three different computations, or ask for a group while the spec returns a
number, or under a title that asks something else. Checking prose against a
spec afterwards only catches the mistakes already thought of.

So the computational part of a generated question is not left to the model.
It is rendered here from the spec, the same way every time: the filters with
their values, the statistic for each group, and exactly what to report. The
model still decides WHAT is asked (the spec) and may add one sentence of
setting. Nothing in the rendered text says which pandas function to use.
"""

from .mess import describe

STAT = {
    "median": "median",
    "mean": "mean",
    "max": "highest value",
    "min": "lowest value",
    "sum": "total",
    "std": "standard deviation",
    "var": "variance",
    "range": "range (highest value minus lowest value)",
}

STAT_SHORT = {
    "median": "median",
    "mean": "mean",
    "max": "maximum",
    "min": "minimum",
    "sum": "total",
    "std": "standard deviation",
    "var": "variance",
    "range": "range",
    "count": "measurement count",
    "nunique": "distinct count",
}

NOUN = {"disease": "disease group", "protein": "protein"}
PLURAL = {"disease": "disease groups", "protein": "proteins"}

OPS = {
    ">=": "{c} is at least {v}",
    ">": "{c} is strictly above {v}",
    "<=": "{c} is at most {v}",
    "<": "{c} is strictly below {v}",
    "==": "{c} is {v}",
    "!=": "{c} is not {v}",
    "in": "{c} is one of {v}",
    "not in": "{c} is none of {v}",
    "notna": "{c} is recorded (not missing)",
    "isna": "{c} is missing",
}

TITLE_MAX = 60

TITLE_ABBR = {"standard deviation": "std dev", "measurement count": "count"}


def _value(v):
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)

    return str(v)


def _group(spec, plural=False):
    names = PLURAL if plural else NOUN

    if len(spec.group_by) == 1:
        return names.get(spec.group_by[0], spec.group_by[0])

    both = " and ".join(spec.group_by)

    return f"combinations of {both}" if plural else f"combination of {both}"


def _filters(spec):
    clauses = [
        OPS[f.op].format(c=f.column, v=_value(f.value)) for f in spec.filters
    ]

    if not clauses:
        return ""

    return "Use only rows where " + " and ".join(clauses) + "."


def _statistic(spec):
    group = _group(spec)

    if spec.agg == "count":
        return f"For each {group}, count the measurements."

    if spec.agg == "nunique":
        return (
            f"For each {group}, count the distinct {spec.value} values "
            "(each different value counted once)."
        )

    return f"For each {group}, compute the {STAT[spec.agg]} of {spec.value}."


def _report(spec):
    noun, plural = _group(spec), _group(spec, plural=True)

    return {
        "idxmax": f"Report which {noun} has the highest of these values.",
        "idxmin": f"Report which {noun} has the lowest of these values.",
        "max": "Report the highest of these values.",
        "min": "Report the lowest of these values.",
        "diff_extremes": (
            "Report the difference between the highest and the lowest of "
            f"these values across {plural}."
        ),
        "nlargest": (
            f"Return the {spec.n} {plural} with the highest of these values, "
            "as a Series ordered from highest to lowest."
        ),
        "nsmallest": (
            f"Return the {spec.n} {plural} with the lowest of these values, "
            "as a Series ordered from lowest to highest."
        ),
        "none": f"Return these values for every {noun}, as a Series.",
    }[spec.reduce]


def render_mission(spec, kinds=(), level=1, context=""):
    """
    The question text. `kinds` are the data defects of a cleaning topic: named
    at level 1, only hinted at above it, where finding them is the exercise.
    """
    parts = []

    if context.strip():
        parts.append(context.strip())

    if kinds:
        if level <= 1:
            parts.append(
                "The raw export has these problems, which you must fix first: "
                + describe(kinds).replace("  - ", "").replace("\n", "; ")
                + "."
            )
        else:
            parts.append(
                "This is a raw export that needs cleaning before you analyse it."
            )

    scope = _filters(spec)

    if scope:
        parts.append(scope)

    parts.append(_statistic(spec))
    parts.append(_report(spec))

    return " ".join(parts)


def _title_group(spec, plural, short):
    if len(spec.group_by) == 1:
        return _group(spec, plural=plural)

    if short:
        return "pairs" if plural else "pair"

    return "disease-protein pairs" if plural else "disease-protein pair"


def _title_forms(spec):
    """Wordings from fullest to shortest; the first that fits is used."""
    stat = STAT_SHORT[spec.agg]
    abbreviated = TITLE_ABBR.get(stat, stat)

    for stat_label, short in ((stat, False), (abbreviated, False), (abbreviated, True)):
        if spec.agg in ("count", "nunique"):
            what = stat_label
        else:
            what = f"{stat_label} of {spec.value}"

        group = _title_group(spec, plural=False, short=short)
        plural = _title_group(spec, plural=True, short=short)

        yield {
            "idxmax": f"Highest {what} by {group}",
            "idxmin": f"Lowest {what} by {group}",
            "max": f"Highest group {what}",
            "min": f"Lowest group {what}",
            "diff_extremes": f"Gap in {what} across {plural}",
            "nlargest": f"Top {spec.n} {plural} by {what}",
            "nsmallest": f"Bottom {spec.n} {plural} by {what}",
            "none": f"{what.capitalize()} per {group}",
        }[spec.reduce]


def render_title(spec):
    """A title that states the question, never truncated mid-word."""
    forms = list(_title_forms(spec))

    for form in forms:
        if len(form) <= TITLE_MAX:
            return form

    return forms[-1]


def render_challenge(challenge, kinds=()):
    """
    The challenge with its mission and title written from its spec. Whatever
    the model wrote there is replaced; its `context`, if any, is kept as the
    opening sentence.
    """
    return challenge.model_copy(
        update={
            "mission": render_mission(
                challenge.spec, kinds, challenge.level, challenge.context
            ),
            "title": render_title(challenge.spec),
        }
    )
