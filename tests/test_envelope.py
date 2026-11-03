"""The training envelope: inputs out of range, non-numbers and unseen combinations are refused with reasons."""

import numpy as np
import pytest

from earthquake_screening.buildings.dataset import INPUT_NAMES
from earthquake_screening.screening.envelope import TrainingEnvelope

INPUT_COUNT = len(INPUT_NAMES)


def two_clusters(rows_per_cluster: int, seed: int) -> np.ndarray:
    """Rows whose first two inputs are both low (0 to 1) or both high (9 to 10), never mixed; the rest constant."""
    rng = np.random.default_rng(seed)
    low = np.full((rows_per_cluster, INPUT_COUNT), 0.5)
    low[:, :2] = rng.uniform(0.0, 1.0, (rows_per_cluster, 2))
    high = low.copy()
    high[:, :2] += 9.0
    return np.vstack([low, high])


def two_cluster_envelope() -> TrainingEnvelope:
    return TrainingEnvelope.from_training(two_clusters(200, seed=0), two_clusters(100, seed=1))


def test_a_training_like_row_is_accepted() -> None:
    envelope = two_cluster_envelope()
    assert envelope.refusal_reasons(np.full(INPUT_COUNT, 0.5)) == []


def test_an_input_above_its_range_is_refused_with_its_name_and_range() -> None:
    row = np.full(INPUT_COUNT, 0.5)
    row[INPUT_NAMES.index("storeys")] = 30.0
    reasons = two_cluster_envelope().refusal_reasons(row)
    assert len(reasons) == 1
    assert reasons[0].startswith("storeys = 30 is outside the training range")


def test_a_non_number_is_refused() -> None:
    row = np.full(INPUT_COUNT, 0.5)
    row[-1] = np.nan
    assert two_cluster_envelope().refusal_reasons(row) == [f"{INPUT_NAMES[-1]} is nan, not a number"]


def test_an_unseen_combination_is_refused_though_each_input_is_in_range() -> None:
    row = np.full(INPUT_COUNT, 0.5)
    row[1] = 9.5  # first input low, second high: never seen together
    reasons = two_cluster_envelope().refusal_reasons(row)
    assert len(reasons) == 1
    assert "unlike any training case" in reasons[0]


def test_the_distance_limit_lets_99_percent_of_calibration_rows_through() -> None:
    rng = np.random.default_rng(1)
    fit = rng.uniform(0.0, 1.0, (500, INPUT_COUNT))
    calibration = rng.uniform(0.1, 0.9, (300, INPUT_COUNT))  # inside every range, so only the distance decides
    envelope = TrainingEnvelope.from_training(fit, calibration)
    assert np.mean(envelope.accepts(calibration)) == 0.99


def test_checking_many_rows_agrees_with_checking_one() -> None:
    envelope = two_cluster_envelope()
    rows = np.full((4, INPUT_COUNT), 0.5)
    rows[1, 0] = 50.0
    rows[2, 1] = 9.5
    rows[3, 2] = np.inf
    assert envelope.accepts(rows).tolist() == [envelope.refusal_reasons(row) == [] for row in rows]
    assert envelope.accepts(rows).tolist() == [True, False, False, False]


def test_a_row_with_a_missing_input_is_an_error_naming_the_expected_inputs() -> None:
    with pytest.raises(ValueError, match="expected 12 inputs in this order: storeys"):
        two_cluster_envelope().refusal_reasons(np.full(INPUT_COUNT - 1, 0.5))
