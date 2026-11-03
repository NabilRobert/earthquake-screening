"""Active learning: the starting pair, the query rule and the snapshots."""

import numpy as np
import pytest

from earthquake_screening.surrogate.active_learning import run_active_learning, standardised_features, starting_pair


def _separable_pool(count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((count, 2))
    return features, features[:, 0] + 0.5 * features[:, 1] > 0.8


def test_starting_pair_is_one_mild_survivor_and_one_strong_failure() -> None:
    rng = np.random.default_rng(0)
    pga = rng.uniform(1, 10, 500)
    linear_peak = pga * rng.uniform(0.9, 1.1, 500)
    has_failed = linear_peak > 7
    mild, strong = starting_pair(pga, linear_peak, has_failed, rng)
    assert pga[mild] < np.median(pga) and not has_failed[mild]
    assert pga[strong] > np.quantile(pga, 0.9) and has_failed[strong]


def test_starting_pair_fails_clearly_when_no_strong_record_failed() -> None:
    pga = np.linspace(1, 10, 100)
    with pytest.raises(ValueError, match="failure=True"):
        starting_pair(pga, pga, np.zeros(100, dtype=bool), np.random.default_rng(0))


def test_each_query_is_the_unlabelled_record_closest_to_the_hyperplane() -> None:
    features, has_failed = _separable_pool(300)
    start = np.array([int(np.argmin(features[:, 0])), int(np.argmax(features[:, 0]))])
    snapshots = run_active_learning(features, has_failed, start, budgets=(2, 3))
    queried = snapshots[3].labelled_indices[-1]
    unlabelled = np.setdiff1d(np.arange(300), start)
    assert queried == unlabelled[np.argmin(np.abs(snapshots[2].scores[unlabelled]))]


def test_a_snapshot_is_taken_at_each_budget() -> None:
    features, has_failed = _separable_pool(300)
    start = np.array([int(np.argmin(features[:, 0])), int(np.argmax(features[:, 0]))])
    snapshots = run_active_learning(features, has_failed, start, budgets=(10, 25))
    assert {budget: snapshot.labelled_indices.size for budget, snapshot in snapshots.items()} == {10: 10, 25: 25}
    assert len(set(snapshots[25].labelled_indices.tolist())) == 25


def test_features_are_box_cox_transformed_and_standardised() -> None:
    skewed = np.random.default_rng(0).lognormal(size=(1000, 3))
    standardised = standardised_features(skewed)
    assert standardised.mean(axis=0) == pytest.approx([0, 0, 0], abs=1e-9)
    assert standardised.std(axis=0) == pytest.approx([1, 1, 1], abs=1e-9)
