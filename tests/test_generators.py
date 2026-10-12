"""
The dataset has to be one where mistakes show.
"""

import pytest

from morning_drill.generators import generate_biomarker_dataset
from scripts.discrimination_yield import measure


def test_the_same_seed_gives_the_same_table():
    a = generate_biomarker_dataset(n=2_000, seed=5)
    b = generate_biomarker_dataset(n=2_000, seed=5)

    assert a.equals(b)


def test_version_one_is_still_available_and_unchanged():
    a = generate_biomarker_dataset(n=2_000, seed=5, version=1)

    assert list(a.columns) == ["sample_id", "disease", "protein", "intensity", "qc_score"]
    assert generate_biomarker_dataset(n=2_000, seed=5, version=1).equals(a)
    assert not a.equals(generate_biomarker_dataset(n=2_000, seed=5, version=2))


def test_the_columns_and_categories_match_the_schema():
    from morning_drill.spec import DATASETS

    df = generate_biomarker_dataset(n=3_000, seed=1)
    schema = DATASETS["biomarker"]

    assert list(df.columns) == list(schema["columns"])
    assert set(df.disease) == set(schema["categories"]["disease"])
    assert set(df.protein) == set(schema["categories"]["protein"])


def test_each_protein_has_a_different_winner_and_pooling_hides_it():
    df = generate_biomarker_dataset(n=30_000, seed=3)
    good = df[df.qc_score >= 0.7]

    by_protein = {
        p: g.groupby("disease").intensity.median().idxmax()
        for p, g in good.groupby("protein")
    }
    pooled = good.groupby("disease").intensity.median().idxmax()

    assert len(set(by_protein.values())) >= 3
    assert pooled not in by_protein.values()


def test_the_mean_and_the_median_name_different_groups_somewhere():
    df = generate_biomarker_dataset(n=30_000, seed=3)
    good = df[df.qc_score >= 0.7]

    differs = [
        p
        for p, g in good.groupby("protein")
        if g.groupby("disease").intensity.mean().idxmax()
        != g.groupby("disease").intensity.median().idxmax()
    ]

    assert len(differs) >= 2


def test_dropping_the_qc_filter_changes_a_winner():
    df = generate_biomarker_dataset(n=30_000, seed=3)

    clean = df[df.qc_score >= 0.85].groupby("disease").intensity.median().idxmax()
    raw = df.groupby("disease").intensity.median().idxmax()

    assert clean != raw


def test_questions_tell_right_from_wrong_more_often_than_before():
    """
    The measure the generator was tuned against: of random specs, the share in
    which some wrong statistic or dropped filter changes the answer.
    """
    old = measure(n_specs=80, version=1)
    new = measure(n_specs=80, version=2)

    assert new["discriminating"] >= old["discriminating"] + 0.08
    assert new["fully_discriminating"] > old["fully_discriminating"]
