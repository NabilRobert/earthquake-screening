"""The baseline a surrogate must beat: one drift curve per Hazus height class, Hazus-style.

Hazus sorts steel moment frames (S1) into low (1-3 storeys), mid (4-7) and high (8+)
and treats every building in a class alike. Here each class gets one line,
ln drift = a + b * ln Sa(T), fitted on the same rows as the surrogate, with T the one
spectral period that fits that class best. It knows nothing else about a frame.
pelicun's Hazus library holds damage-state limits, not drift-demand curves, so the
lines are fitted here (docs/building-surrogate/prj2048.md).
"""

import math
from dataclasses import dataclass

import numpy as np

from earthquake_screening.buildings.dataset import INPUT_NAMES, SPECTRAL_PERIODS_S, STOREYS_COLUMN, FloatArray

# (class, fewest storeys, most storeys), from Hazus's building-type table for S1.
HAZUS_S1_HEIGHT_CLASSES = (("low", 1, 3), ("mid", 4, 7), ("high", 8, math.inf))
SPECTRAL_COLUMNS = tuple(INPUT_NAMES.index(f"ln_sa_{period:g}s_g") for period in SPECTRAL_PERIODS_S)


def height_class(storeys: float) -> str:
    for name, fewest, most in HAZUS_S1_HEIGHT_CLASSES:
        if fewest <= storeys <= most:
            return name
    raise ValueError(f"{storeys} storeys fits no Hazus height class: expected a whole number of at least 1")


@dataclass(frozen=True)
class ClassLine:
    spectral_column: int
    intercept: float
    slope: float


def _fit_line(ln_sa: FloatArray, ln_drifts: FloatArray) -> tuple[float, float, float]:
    """(intercept, slope, RMSE) of the least-squares line."""
    coefficients: FloatArray = np.polyfit(ln_sa, ln_drifts, deg=1)
    slope, intercept = float(coefficients[0]), float(coefficients[1])
    residuals = ln_drifts - (intercept + slope * ln_sa)
    return intercept, slope, float(np.sqrt(np.mean(residuals**2)))


def _best_line(inputs: FloatArray, ln_drifts: FloatArray) -> ClassLine:
    best: tuple[float, ClassLine] | None = None
    for column in SPECTRAL_COLUMNS:
        intercept, slope, error = _fit_line(inputs[:, column], ln_drifts)
        if best is None or error < best[0]:
            best = (error, ClassLine(column, intercept, slope))
    assert best is not None  # SPECTRAL_COLUMNS is never empty
    return best[1]


@dataclass(frozen=True)
class ClassBaseline:
    lines: dict[str, ClassLine]

    @classmethod
    def fit(cls, inputs: FloatArray, ln_drifts: FloatArray) -> "ClassBaseline":
        classes = np.array([height_class(storeys) for storeys in inputs[:, STOREYS_COLUMN]])
        lines = {name: _best_line(inputs[classes == name], ln_drifts[classes == name]) for name in set(classes)}
        return cls(lines)

    def predict(self, inputs: FloatArray) -> FloatArray:
        predicted = np.empty(inputs.shape[0], dtype=np.float64)
        for row, values in enumerate(inputs):
            line = self.lines[height_class(values[STOREYS_COLUMN])]
            predicted[row] = line.intercept + line.slope * values[line.spectral_column]
        return predicted
