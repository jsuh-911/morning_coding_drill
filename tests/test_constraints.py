from morning_drill.constraints import (
    analyse_code,
    check_constraints,
    looks_hardcoded,
)


def test_detects_for_loop():
    violations, _ = check_constraints("for i in range(3):\n    pass", ["no_loops"])
    assert violations


def test_detects_comprehension():
    violations, _ = check_constraints("answer = [x for x in df]", ["no_loops"])
    assert violations


def test_detects_apply():
    violations, _ = check_constraints(
        'answer = df["a"].apply(lambda v: v)', ["no_apply"]
    )
    assert violations


def test_detects_iterrows():
    violations, _ = check_constraints(
        "answer = [r for _, r in df.iterrows()]", ["no_iterrows"]
    )
    assert violations


def test_vectorized_solution_is_clean():
    violations, _ = check_constraints(
        'answer = df.groupby("disease")["intensity"].median().idxmax()',
        ["no_loops", "no_apply", "no_iterrows", "use_pandas"],
    )
    assert violations == []


def test_syntax_error_is_reported_not_raised():
    violations, facts = check_constraints("answer = (", ["no_loops"])
    assert facts.syntax_error
    assert "does not parse" in violations[0]


def test_empty_submission_does_not_crash():
    violations, facts = check_constraints("", ["no_loops", "use_pandas"])
    assert isinstance(violations, list)


def test_hardcoded_string_answer_is_caught():
    assert looks_hardcoded(analyse_code('answer = "AD"'), "AD")


def test_hardcoded_number_answer_is_caught():
    assert looks_hardcoded(analyse_code("answer = 11.19"), 11.19)


def test_real_work_mentioning_a_threshold_is_not_accused():
    code = 'answer = df.loc[df["qc_score"] >= 0.85].groupby("d")["i"].median().idxmax()'
    assert not looks_hardcoded(analyse_code(code), 0.85)
