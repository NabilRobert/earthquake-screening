"""The training envelope: refuse to answer for inputs the surrogate never learned from.

Two checks, both computed from training data and saved with the model (ADR-009):
- each input must lie within its range on the fit rows;
- the whole input row must be near some fit row: its distance to the nearest one, in
  standardised inputs, at most the 99th percentile of the calibration rows' distances.
  Each input can be in range while the combination is new; this catches that.
An input outside the envelope gets reasons, never a number. Nothing here imports
OpenSeesPy: screening code must run without it (ADR-002).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from earthquake_screening.buildings.dataset import INPUT_NAMES, FloatArray

ACCEPTED_CALIBRATION_PERCENTILE = 99.0


@dataclass(frozen=True)
class TrainingEnvelope:
    minimum: FloatArray
    maximum: FloatArray
    mean: FloatArray
    scale: FloatArray
    neighbours: NearestNeighbors  # fitted on the standardised fit rows
    distance_limit: float

    @classmethod
    def from_training(cls, fit_inputs: FloatArray, calibration_inputs: FloatArray) -> "TrainingEnvelope":
        mean = fit_inputs.mean(axis=0)
        # A constant input would divide by zero; its range check still pins it.
        scale = np.where(fit_inputs.std(axis=0) > 0, fit_inputs.std(axis=0), 1.0)
        neighbours = NearestNeighbors(n_neighbors=1).fit((fit_inputs - mean) / scale)
        distances, _ = neighbours.kneighbors((calibration_inputs - mean) / scale)
        return cls(
            minimum=fit_inputs.min(axis=0),
            maximum=fit_inputs.max(axis=0),
            mean=mean,
            scale=scale,
            neighbours=neighbours,
            distance_limit=float(np.percentile(distances, ACCEPTED_CALIBRATION_PERCENTILE)),
        )

    def _nearest_distances(self, inputs: FloatArray) -> FloatArray:
        distances, _ = self.neighbours.kneighbors((inputs - self.mean) / self.scale)
        return distances[:, 0]

    def accepts(self, inputs: FloatArray) -> NDArray[np.bool_]:
        """For many rows at once: True where every check passes."""
        is_finite = np.all(np.isfinite(inputs), axis=1)
        safe_inputs = np.where(np.isfinite(inputs), inputs, self.mean)
        is_in_range = np.all((safe_inputs >= self.minimum) & (safe_inputs <= self.maximum), axis=1)
        return is_finite & is_in_range & (self._nearest_distances(safe_inputs) <= self.distance_limit)

    def refusal_reasons(self, inputs: FloatArray) -> list[str]:
        """Why one input row is refused; an empty list means it is accepted."""
        if inputs.shape != (len(INPUT_NAMES),):
            raise ValueError(
                f"expected {len(INPUT_NAMES)} inputs in this order: {', '.join(INPUT_NAMES)}; got {inputs.shape}"
            )
        reasons: list[str] = []
        for name, value, low, high in zip(INPUT_NAMES, inputs, self.minimum, self.maximum, strict=True):
            if not np.isfinite(value):
                reasons.append(f"{name} is {value}, not a number")
            elif not low <= value <= high:
                reasons.append(f"{name} = {value:g} is outside the training range {low:g} to {high:g}")
        if reasons:
            return reasons
        distance = float(self._nearest_distances(inputs[np.newaxis, :])[0])
        if distance > self.distance_limit:
            reasons.append(
                f"this combination of inputs is unlike any training case (distance {distance:.2f}, "
                f"limit {self.distance_limit:.2f}), although each input is within its range"
            )
        return reasons
