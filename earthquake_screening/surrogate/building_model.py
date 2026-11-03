"""The two candidate surrogates for a frame's peak drift, and how one is chosen.

Both predict ln(peak inter-storey drift) from INPUT_NAMES. They are compared by
cross-validation on the fit rows only, grouped by frame so a frame's rows never sit
on both sides of a fold. Neither is tuned: the settings below were fixed before any
run (docs/building-surrogate/prj2048.md).
"""

from typing import Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from earthquake_screening.buildings.dataset import FloatArray, IntArray

BOOSTING = "gradient boosting"
GAUSSIAN_PROCESS = "gaussian process"
CANDIDATES = (BOOSTING, GAUSSIAN_PROCESS)
BOOSTING_ITERATIONS = 400
BOOSTING_LEARNING_RATE = 0.05
# A GP's fit costs n^3; 2,000 rows is the most a laptop handles in about a minute.
GP_SAMPLE_ROWS = 2000
CROSS_VALIDATION_FOLDS = 5


class Regressor(Protocol):
    def predict(self, X: FloatArray) -> FloatArray: ...


def _gaussian_process(input_count: int, seed: int) -> Pipeline:
    kernel = ConstantKernel() * RBF(length_scale=np.ones(input_count)) + WhiteKernel()
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed))


def fit_candidate(name: str, inputs: FloatArray, ln_drifts: FloatArray, seed: int) -> Regressor:
    """Fit one candidate. The GP sees a random GP_SAMPLE_ROWS of the rows, the boosting all of them."""
    if name == BOOSTING:
        model = HistGradientBoostingRegressor(
            max_iter=BOOSTING_ITERATIONS, learning_rate=BOOSTING_LEARNING_RATE, early_stopping=False, random_state=seed
        )
        return model.fit(inputs, ln_drifts)
    if name == GAUSSIAN_PROCESS:
        rows = np.random.default_rng(seed).permutation(ln_drifts.size)[:GP_SAMPLE_ROWS]
        return _gaussian_process(inputs.shape[1], seed).fit(inputs[rows], ln_drifts[rows])
    raise ValueError(f"unknown candidate {name!r}: expected one of {CANDIDATES}")


def rmse(predicted: FloatArray, actual: FloatArray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def cross_validate(inputs: FloatArray, ln_drifts: FloatArray, frames: IntArray, seed: int) -> dict[str, list[float]]:
    """Each candidate's RMSE of ln drift on each held-out fold of frames."""
    scores: dict[str, list[float]] = {name: [] for name in CANDIDATES}
    for train, held_out in GroupKFold(n_splits=CROSS_VALIDATION_FOLDS).split(inputs, ln_drifts, frames):
        for name in CANDIDATES:
            model = fit_candidate(name, inputs[train], ln_drifts[train], seed)
            scores[name].append(rmse(model.predict(inputs[held_out]), ln_drifts[held_out]))
    return scores


def choose(scores: dict[str, list[float]]) -> str:
    """The candidate with the lower mean RMSE; a tie goes to boosting, the faster one."""
    return min(CANDIDATES, key=lambda name: (float(np.mean(scores[name])), name != BOOSTING))
