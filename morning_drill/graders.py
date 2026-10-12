import numpy as np
import pandas as pd


CONTAINER_TYPES = (
    pd.DataFrame,
    pd.Series,
    pd.Index,
    np.ndarray,
    list,
    tuple,
    set,
    dict,
)


def compare_answers(answer, expected):
    """
    General-purpose answer comparison.

    Always returns a plain bool. A learner who submits a Series where a
    scalar was wanted gets False, not a ValueError from an elementwise
    comparison leaking into an `if`.
    """
    # A two-column group key: idxmax on a MultiIndex returns a tuple. The
    # container rule below is for a Series or list where a scalar was wanted,
    # and would otherwise reject the right answer.
    if isinstance(expected, tuple):
        if not isinstance(answer, (tuple, list)) or len(answer) != len(expected):
            return False

        return all(compare_answers(a, e) for a, e in zip(answer, expected))

    if isinstance(expected, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(
                answer.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_dtype=False,
            )
            return True
        except (AssertionError, AttributeError, TypeError, ValueError):
            return False

    if isinstance(expected, pd.Series):
        try:
            pd.testing.assert_series_equal(
                answer.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_dtype=False,
                check_names=False,
            )
            return True
        except (AssertionError, AttributeError, TypeError, ValueError):
            return False

    # expected is a scalar: no container can equal it
    if isinstance(answer, CONTAINER_TYPES):
        return False

    # Cleaning pipelines reorder floating-point work; two routes to the same
    # number can differ in the last bit.
    if (
        isinstance(expected, (int, float, np.number))
        and not isinstance(expected, (bool, np.bool_))
        and isinstance(answer, (int, float, np.number))
        and not isinstance(answer, (bool, np.bool_))
    ):
        return bool(np.isclose(answer, expected, rtol=1e-9, atol=0, equal_nan=True))

    try:
        result = answer == expected
    except (TypeError, ValueError):
        return False

    if isinstance(result, (bool, np.bool_)):
        return bool(result)

    # anything still elementwise is a shape mismatch, i.e. wrong
    return False


def _describe(obj):
    """
    Structural description of an object. Deliberately value-free.
    """
    desc = {"type": type(obj).__name__}

    if isinstance(obj, pd.DataFrame):
        desc["shape"] = list(obj.shape)
        desc["columns"] = list(obj.columns)
        desc["index_name"] = obj.index.name
    elif isinstance(obj, pd.Series):
        desc["length"] = len(obj)
        desc["dtype"] = str(obj.dtype)
        desc["index_name"] = obj.index.name
    elif isinstance(obj, (np.ndarray, list, tuple, set)):
        desc["length"] = len(obj)
    elif isinstance(obj, dict):
        desc["length"] = len(obj)
    else:
        desc["scalar"] = True
        desc["python_type"] = type(obj).__module__ + "." + type(obj).__name__

    return desc


def discrepancy_report(answer, expected):
    """
    Describe how the submitted object differs in *shape* from the expected
    one, without disclosing either value.

    This is what the coach is allowed to see on a first attempt. It can say
    "you returned a Series where a single value was wanted" and cannot say
    which value that is.
    """
    got = _describe(answer)
    want = _describe(expected)

    notes = []

    if got["type"] != want["type"]:
        notes.append(
            f"submitted a {got['type']} where a {want['type']} was expected"
        )

    if want.get("scalar") and not got.get("scalar"):
        n = got.get("length") or got.get("shape")
        notes.append(
            f"a single value was expected but a collection of {n} was submitted; "
            "an aggregation or a reduction step is probably missing"
        )

    if got.get("scalar") and not want.get("scalar"):
        notes.append(
            "a collection was expected but a single value was submitted"
        )

    if "shape" in got and "shape" in want and got["shape"] != want["shape"]:
        notes.append(
            f"shape {got['shape']} does not match the expected {want['shape']}"
        )

    if "columns" in got and "columns" in want:
        missing = [c for c in want["columns"] if c not in got["columns"]]
        extra = [c for c in got["columns"] if c not in want["columns"]]
        if missing:
            notes.append(f"missing columns: {missing}")
        if extra:
            notes.append(f"unexpected columns: {extra}")

    if "length" in got and "length" in want and got["length"] != want["length"]:
        notes.append(
            f"{got['length']} elements submitted, {want['length']} expected"
        )

    if not notes:
        notes.append(
            "the result has the right type and shape but the wrong value, "
            "so the pipeline structure is sound and a step computes the "
            "wrong thing"
        )

    return {
        "submitted": got,
        "expected_structure": want,
        "notes": notes,
    }
