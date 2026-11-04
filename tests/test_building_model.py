"""Choosing between gradient boosting and a Gaussian process, by cross-validation on the fit rows."""

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline

from earthquake_screening.surrogate import building_model
from earthquake_screening.surrogate.building_model import BOOSTING, GAUSSIAN_PROCESS, choose, fit_candidate


def test_choose_takes_the_lower_mean_error() -> None:
    assert choose({BOOSTING: [0.5, 0.6], GAUSSIAN_PROCESS: [0.4, 0.5]}) == GAUSSIAN_PROCESS


def test_a_tie_goes_to_boosting() -> None:
    assert choose({BOOSTING: [0.5, 0.5], GAUSSIAN_PROCESS: [0.4, 0.6]}) == BOOSTING


def test_gaussian_process_trains_on_at_most_its_sample_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(building_model, "GP_SAMPLE_ROWS", 30)
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(200, 3))
    model = fit_candidate(GAUSSIAN_PROCESS, inputs, inputs[:, 0], seed=0)
    assert isinstance(model, Pipeline)
    gaussian_process = model.steps[-1][1]
    assert isinstance(gaussian_process, GaussianProcessRegressor)
    assert gaussian_process.X_train_.shape[0] == 30


def test_an_unknown_candidate_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown candidate"):
        fit_candidate("neural network", np.zeros((4, 2)), np.zeros(4), seed=0)
