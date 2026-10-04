"""
Static analysis of a submission.

The grader agent reviews reasoning and idiom. The checks here are the ones a
model should never be asked to do, because they are decidable: did a for-loop
appear, was .apply() used where a vectorized operation exists, was the answer
simply typed in as a literal.
"""

import ast


class CodeFacts:
    def __init__(self):
        self.syntax_error = None
        self.has_for = False
        self.has_while = False
        self.has_comprehension = False
        self.methods = []
        self.string_literals = []
        self.number_literals = []
        self.names = set()

    @property
    def has_loop(self):
        return self.has_for or self.has_while

    def __repr__(self):
        return (
            f"CodeFacts(loop={self.has_loop}, methods={self.methods[:8]}, "
            f"syntax_error={self.syntax_error!r})"
        )


def analyse_code(code):
    """
    Facts about a submission's source. Never executes it.
    """
    facts = CodeFacts()

    if not code or not code.strip():
        return facts

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        facts.syntax_error = f"line {exc.lineno}: {exc.msg}"
        return facts

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            facts.has_for = True
        elif isinstance(node, ast.While):
            facts.has_while = True
        elif isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.DictComp)):
            facts.has_comprehension = True
        elif isinstance(node, ast.Attribute):
            facts.methods.append(node.attr)
        elif isinstance(node, ast.Name):
            facts.names.add(node.id)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                facts.string_literals.append(node.value)
            elif isinstance(node.value, (int, float)):
                facts.number_literals.append(node.value)

    return facts


def looks_hardcoded(facts, expected):
    """
    Was the answer simply typed in?

    True only when the expected answer appears as a literal AND no pandas
    work was done, so a legitimate solution that happens to mention a
    threshold is not accused.
    """
    did_pandas_work = any(
        m in facts.methods
        for m in (
            "groupby",
            "loc",
            "iloc",
            "agg",
            "median",
            "mean",
            "sum",
            "max",
            "min",
            "idxmax",
            "idxmin",
            "nlargest",
            "nsmallest",
            "query",
            "merge",
            "pivot_table",
            "value_counts",
        )
    )

    if did_pandas_work:
        return False

    if isinstance(expected, str):
        return expected in facts.string_literals

    try:
        return any(
            abs(float(n) - float(expected)) < 1e-9
            for n in facts.number_literals
        )
    except (TypeError, ValueError):
        return False


# constraint id -> (text shown to the student, predicate on CodeFacts)
CONSTRAINT_CHECKS = {
    "no_loops": (
        "Do not use a for-loop or a comprehension; vectorize instead",
        lambda f: not f.has_loop and not f.has_comprehension,
    ),
    "no_apply": (
        "Use a vectorized operation rather than .apply()",
        lambda f: "apply" not in f.methods,
    ),
    "no_iterrows": (
        "Do not iterate over rows",
        lambda f: not any(
            m in f.methods for m in ("iterrows", "itertuples", "iteritems")
        ),
    ),
    "use_pandas": (
        "Use pandas",
        lambda f: bool(f.methods),
    ),
}


def check_constraints(code, constraint_ids):
    """
    Returns (violations, facts). A violation is a human-readable string.
    """
    facts = analyse_code(code)

    if facts.syntax_error:
        return [f"the code does not parse ({facts.syntax_error})"], facts

    violations = []

    for cid in constraint_ids:
        check = CONSTRAINT_CHECKS.get(cid)

        if check is None:
            continue

        text, predicate = check

        if not predicate(facts):
            violations.append(text)

    return violations, facts
