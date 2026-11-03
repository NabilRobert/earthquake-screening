"""Slice 2's measures and pass rule, and one run end to end on made-up data."""

import dataclasses
import math

import numpy as np
import pytest

from earthquake_screening.buildings.experiment import (
    Accuracy,
    Evaluation,
    ExperimentResult,
    accuracy,
    has_passed,
    run_experiment,
)
from earthquake_screening.buildings.report import format_report
from earthquake_screening.buildings.split import split_prj2048
from earthquake_screening.surrogate.building_model import CANDIDATES
from tests.conftest import made_up_prj2048


def test_accuracy_on_a_hand_worked_case() -> None:
    actual = np.log(np.array([0.01, 0.01, 0.01, 0.01]))
    predicted = actual + np.log(np.array([1.0, 1.2, 3.0, 1 / 1.6]))
    low, high = actual - 0.1, actual + 0.1
    low[2] = actual[2] + 0.01  # one interval sits above its simulated value
    high[3] = actual[3] - 0.01  # and one below
    measured = accuracy(predicted, actual, (low, high))
    assert measured.within_factor_1_5 == 0.5
    assert measured.within_factor_2 == 0.75
    assert measured.median_relative_error == pytest.approx((0.2 + 0.375) / 2)
    assert measured.coverage == 0.5
    assert measured.rmse_ln == pytest.approx(
        math.sqrt((math.log(1.2) ** 2 + math.log(3) ** 2 + math.log(1.6) ** 2) / 4)
    )


def judged(coverage: float, model_error: float, baseline_error: float, is_clean: bool = True) -> ExperimentResult:
    """A result with only the judged numbers filled in."""
    model = Accuracy(rows=100, rmse_ln=model_error, median_relative_error=0.2, within_factor_1_5=0.8,
                     within_factor_2=0.9, coverage=coverage, median_interval_factor=1.8)  # fmt: skip
    baseline = dataclasses.replace(model, rmse_ln=baseline_error, coverage=None, median_interval_factor=None)
    evaluation = Evaluation(model, baseline, model, baseline, model, 0.01, {1: model}, 0.001)
    split = split_prj2048(made_up_prj2048())
    return ExperimentResult(split, is_clean, {name: [0.5] for name in CANDIDATES}, CANDIDATES[0], evaluation)


@pytest.mark.parametrize(("coverage", "is_pass"), [(0.879, False), (0.88, True), (0.92, True), (0.921, False)])
def test_coverage_must_be_within_two_points_of_90(coverage: float, is_pass: bool) -> None:
    assert has_passed(judged(coverage, model_error=0.3, baseline_error=0.5)) is is_pass


def test_matching_the_baseline_is_not_beating_it() -> None:
    assert not has_passed(judged(0.9, model_error=0.5, baseline_error=0.5))


def test_a_leaky_split_fails_whatever_the_numbers() -> None:
    assert not has_passed(judged(0.9, model_error=0.1, baseline_error=0.5, is_clean=False))


def test_report_says_miss_when_the_rule_fails() -> None:
    report = format_report(judged(0.85, model_error=0.3, baseline_error=0.5), "abc123", run_minutes=1.0)
    assert "## Verdict: MISS" in report
    assert "`abc123`" in report


def test_a_run_on_made_up_data_goes_end_to_end() -> None:
    model, result = run_experiment(made_up_prj2048(motion_count=20), seed=0)
    assert result.is_split_clean
    assert result.chosen in CANDIDATES
    test_coverage = result.evaluation.test.coverage
    assert test_coverage is not None and 0.0 <= test_coverage <= 1.0
    # The made-up drift follows Sa(1 s) and storeys, which the surrogate sees and the baseline half sees.
    assert result.evaluation.test.rmse_ln < 1.0
    assert model.envelope.refusal_reasons(np.full(12, np.nan))
