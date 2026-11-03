"""Fragility estimates and the Delta L2 error (Eq. 37)."""

import numpy as np
import pytest

from earthquake_screening.surrogate.fragility import failure_probabilities, fragility_error, group_by_value


def test_fragility_error_matches_a_hand_calculation() -> None:
    # Group 0: 2 records, rate 0.5, estimate 0.3. Group 1: 2 records, rate 1.0, estimate 0.9.
    # sqrt((2 * 0.2^2 + 2 * 0.1^2) / 4) = sqrt(0.025)
    groups = np.array([0, 0, 1, 1])
    has_failed = np.array([True, False, True, True])
    estimated = np.array([0.3, 0.3, 0.9, 0.9])
    assert fragility_error(estimated, has_failed, groups) == pytest.approx(np.sqrt(0.025))


def test_estimates_equal_to_the_group_rates_have_no_error() -> None:
    groups = np.array([0, 0, 1, 1, 1, 1])
    has_failed = np.array([True, False, True, True, True, False])
    estimated = np.array([0.5, 0.5, 0.75, 0.75, 0.75, 0.75])
    assert fragility_error(estimated, has_failed, groups) == pytest.approx(0.0)


def test_probability_of_failure_rises_with_the_score() -> None:
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(2000)
    has_failed = scores + 0.5 * rng.standard_normal(2000) > 0
    probabilities = failure_probabilities(scores, np.arange(500), has_failed)
    order = np.argsort(scores)
    assert np.all(np.diff(probabilities[order]) >= 0)
    assert probabilities[order[0]] < 0.05 < 0.95 < probabilities[order[-1]]


def test_groups_hold_neighbouring_values() -> None:
    values = np.concatenate([np.full(10, 1.0), np.full(10, 5.0), np.full(10, 9.0)])
    groups = group_by_value(values, group_count=3, seed=0)
    assert len(set(groups[:10])) == len(set(groups[10:20])) == len(set(groups[20:])) == 1
    assert len(set(groups.tolist())) == 3
