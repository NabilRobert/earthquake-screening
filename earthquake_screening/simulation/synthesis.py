"""Draw new parameter sets around the fitted real records (Sainct et al. 2020, section 2.3).

A Gaussian kernel density over the fitted 8-parameter vectors: pick a real record at
random, then add Gaussian noise whose covariance is the sample covariance times a
bandwidth factor. The paper's bandwidth follows Kristan et al.; we use scipy's Scott
factor, a documented deviation (docs/reproduction/sainct2020.md).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from scipy import stats

from earthquake_screening.simulation.ground_motion import FloatArray, GroundMotionParameters

# Noise can push a draw out of the model's valid range (a negative damping, T2 before
# T1). Such draws are redrawn; this caps how many tries one sample may take.
MAX_DRAWS_PER_SAMPLE = 1000


@dataclass(frozen=True)
class FittedRecord:
    record_id: str
    parameters: GroundMotionParameters
    duration_s: float


def save_fitted(records: list[FittedRecord], path: Path) -> None:
    rows = [
        {"record_id": record.record_id, "duration_s": record.duration_s, "parameters": record.parameters.model_dump()}
        for record in records
    ]
    path.write_text(json.dumps(rows, indent=1))


def load_fitted(path: Path) -> list[FittedRecord]:
    rows = json.loads(path.read_text())
    return [
        FittedRecord(row["record_id"], GroundMotionParameters.model_validate(row["parameters"]), row["duration_s"])
        for row in rows
    ]


@dataclass(frozen=True)
class SyntheticRecordSpec:
    """Everything needed to simulate one synthetic record repeatably."""

    parameters: GroundMotionParameters
    # The model's frequency drift is defined over the record's length (Eq. 2), and the
    # paper gives no length for synthetic records, so each one keeps its parent's.
    duration_s: float
    parent_record_id: str
    seed: int


def kernel_covariance(fitted: list[FittedRecord]) -> FloatArray:
    vectors = np.array([record.parameters.as_vector() for record in fitted])
    return np.atleast_2d(stats.gaussian_kde(vectors.T, bw_method="scott").covariance)


def _draw_one(
    fitted: list[FittedRecord], covariance: FloatArray, rng: np.random.Generator
) -> tuple[FittedRecord, GroundMotionParameters]:
    for _ in range(MAX_DRAWS_PER_SAMPLE):
        parent = fitted[int(rng.integers(len(fitted)))]
        vector = rng.multivariate_normal(parent.parameters.as_vector(), covariance)
        try:
            return parent, GroundMotionParameters.from_vector(vector)
        except ValidationError:
            continue
    raise RuntimeError(
        f"{MAX_DRAWS_PER_SAMPLE} draws in a row were invalid: the fitted parameters are too spread out "
        "for this bandwidth. Check the fits for outliers."
    )


def draw_synthetic_specs(fitted: list[FittedRecord], count: int, seed: int) -> list[SyntheticRecordSpec]:
    """count synthetic record specs, each with its own seed, all repeatable from one seed."""
    if len(fitted) < 2:
        raise ValueError(f"need at least 2 fitted records for a density, got {len(fitted)}")
    covariance = kernel_covariance(fitted)
    rng = np.random.default_rng(seed)
    specs: list[SyntheticRecordSpec] = []
    for _ in range(count):
        parent, parameters = _draw_one(fitted, covariance, rng)
        specs.append(
            SyntheticRecordSpec(parameters, parent.duration_s, parent.record_id, seed=int(rng.integers(2**63)))
        )
    return specs
