"""Slice 2's run: split PRJ-2048, choose and fit the surrogate, calibrate its intervals, and judge it.

The pass rule (docs/building-surrogate/prj2048.md) is judged on the test rows, where
the frames and the shaking are both unseen, with no refusals applied:
the split is clean, the 90% intervals cover 88-92%, and the surrogate's error is
lower than the Hazus-style class baseline's on the same rows.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
from mapie.regression import SplitConformalRegressor

from earthquake_screening.buildings.dataset import STOREYS_COLUMN, FloatArray, Prj2048
from earthquake_screening.buildings.split import Rows, Split, is_split_clean, rows_for, split_prj2048
from earthquake_screening.screening.envelope import TrainingEnvelope
from earthquake_screening.surrogate.building_model import choose, cross_validate, fit_candidate, rmse
from earthquake_screening.surrogate.class_baseline import ClassBaseline

CONFIDENCE_LEVEL = 0.90
COVERAGE_BAND = (0.88, 0.92)  # ADR-008: a miss of more than 2 points fails
TIMED_PREDICTIONS = 50


@dataclass(frozen=True)
class Accuracy:
    rows: int
    rmse_ln: float  # RMSE of ln drift: roughly the typical error as a fraction, for small values
    median_relative_error: float  # median of |predicted / simulated - 1|
    within_factor_1_5: float
    within_factor_2: float
    coverage: float | None = None
    median_interval_factor: float | None = None  # the interval is the prediction times or divided by this


def accuracy(predicted: FloatArray, actual: FloatArray, bounds: tuple[FloatArray, FloatArray] | None) -> Accuracy:
    """Error of ln-drift predictions against the simulated ln drifts."""
    log_ratio = np.abs(predicted - actual)
    coverage = interval_factor = None
    if bounds is not None:
        low, high = bounds
        coverage = float(np.mean((actual >= low) & (actual <= high)))
        interval_factor = float(np.exp(np.median(high - low) / 2))
    return Accuracy(
        rows=int(actual.size),
        rmse_ln=rmse(predicted, actual),
        median_relative_error=float(np.median(np.abs(np.exp(predicted - actual) - 1))),
        within_factor_1_5=float(np.mean(log_ratio <= math.log(1.5))),
        within_factor_2=float(np.mean(log_ratio <= math.log(2.0))),
        coverage=coverage,
        median_interval_factor=interval_factor,
    )


@dataclass(frozen=True)
class BuildingModel:
    """What is saved: the chosen surrogate inside its conformal wrapper, and the envelope."""

    name: str
    conformal: SplitConformalRegressor
    envelope: TrainingEnvelope

    def predict_interval(self, inputs: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """(prediction, low, high) in ln drift."""
        predicted, intervals = self.conformal.predict_interval(inputs)
        return predicted, intervals[:, 0, 0], intervals[:, 1, 0]


@dataclass(frozen=True)
class Evaluation:
    test: Accuracy
    baseline_test: Accuracy
    known_shaking: Accuracy
    baseline_known_shaking: Accuracy
    accepted_test: Accuracy | None  # None if the envelope refused every test row
    refusal_rate: float
    test_by_storeys: dict[int, Accuracy]
    seconds_per_screen: float


@dataclass(frozen=True)
class ExperimentResult:
    split: Split
    is_split_clean: bool
    cross_validation: dict[str, list[float]]
    chosen: str
    evaluation: Evaluation


def has_passed(result: ExperimentResult) -> bool:
    test, baseline = result.evaluation.test, result.evaluation.baseline_test
    is_covered = test.coverage is not None and COVERAGE_BAND[0] <= test.coverage <= COVERAGE_BAND[1]
    return result.is_split_clean and is_covered and test.rmse_ln < baseline.rmse_ln


def train(data: Prj2048, split: Split, seed: int) -> tuple[BuildingModel, dict[str, list[float]], ClassBaseline]:
    fit = rows_for(data, split.fit_frames, split.training_motions)
    calibration = rows_for(data, split.calibration_frames, split.held_out_motions)
    scores = cross_validate(fit.inputs, fit.ln_drifts, fit.frames, seed)
    chosen = choose(scores)
    regressor = fit_candidate(chosen, fit.inputs, fit.ln_drifts, seed)
    conformal = SplitConformalRegressor(regressor, confidence_level=CONFIDENCE_LEVEL, prefit=True)
    conformal.conformalize(calibration.inputs, calibration.ln_drifts)
    envelope = TrainingEnvelope.from_training(fit.inputs, calibration.inputs)
    return BuildingModel(chosen, conformal, envelope), scores, ClassBaseline.fit(fit.inputs, fit.ln_drifts)


def _model_accuracy(model: BuildingModel, rows: Rows) -> Accuracy:
    predicted, low, high = model.predict_interval(rows.inputs)
    return accuracy(predicted, rows.ln_drifts, (low, high))


def _by_storeys(model: BuildingModel, rows: Rows) -> dict[int, Accuracy]:
    storeys = rows.inputs[:, STOREYS_COLUMN]
    by_storeys: dict[int, Accuracy] = {}
    for count in np.unique(storeys):
        chosen = storeys == count
        subset = Rows(rows.inputs[chosen], rows.ln_drifts[chosen], rows.frames[chosen], rows.motions[chosen])
        by_storeys[int(count)] = _model_accuracy(model, subset)
    return by_storeys


def seconds_per_screen(model: BuildingModel, inputs: FloatArray) -> float:
    """Mean time to check and predict one input row, as a user's screen would."""
    started = time.perf_counter()
    for row in inputs[:TIMED_PREDICTIONS]:
        model.envelope.refusal_reasons(row)
        model.predict_interval(row[np.newaxis, :])
    return (time.perf_counter() - started) / min(TIMED_PREDICTIONS, inputs.shape[0])


def evaluate(model: BuildingModel, baseline: ClassBaseline, data: Prj2048, split: Split) -> Evaluation:
    test = rows_for(data, split.test_frames, split.held_out_motions)
    known = rows_for(data, split.test_frames, split.training_motions)
    accepted = model.envelope.accepts(test.inputs)
    kept = Rows(test.inputs[accepted], test.ln_drifts[accepted], test.frames[accepted], test.motions[accepted])
    return Evaluation(
        test=_model_accuracy(model, test),
        baseline_test=accuracy(baseline.predict(test.inputs), test.ln_drifts, None),
        known_shaking=_model_accuracy(model, known),
        baseline_known_shaking=accuracy(baseline.predict(known.inputs), known.ln_drifts, None),
        accepted_test=_model_accuracy(model, kept) if accepted.any() else None,
        refusal_rate=float(1 - accepted.mean()),
        test_by_storeys=_by_storeys(model, test),
        seconds_per_screen=seconds_per_screen(model, test.inputs),
    )


def run_experiment(data: Prj2048, seed: int) -> tuple[BuildingModel, ExperimentResult]:
    split = split_prj2048(data)
    model, scores, baseline = train(data, split, seed)
    result = ExperimentResult(
        split=split,
        is_split_clean=is_split_clean(split),
        cross_validation=scores,
        chosen=model.name,
        evaluation=evaluate(model, baseline, data, split),
    )
    return model, result
