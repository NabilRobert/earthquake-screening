"""Shared test inputs: one plausible set of ground-motion parameters, and made-up fitted records.

The made-up records stand in for real ESM fits so the pipeline can be tested without
downloading anything. They are not data and no result is drawn from them.
"""

import numpy as np
import pytest

from earthquake_screening.simulation.ground_motion import GroundMotionParameters
from earthquake_screening.simulation.synthesis import FittedRecord

TYPICAL = GroundMotionParameters(
    envelope_amplitude=0.8,
    decay_rate=0.4,
    decay_shape=1.2,
    build_up_end_s=2.0,
    strong_phase_end_s=6.0,
    start_frequency=30.0,
    end_frequency=12.0,
    filter_damping=0.3,
)


@pytest.fixture
def typical_parameters() -> GroundMotionParameters:
    return TYPICAL


def made_up_fitted_records(count: int, seed: int) -> list[FittedRecord]:
    """Parameters scattered around TYPICAL, a stand-in for fitted real records."""
    rng = np.random.default_rng(seed)
    records: list[FittedRecord] = []
    for index in range(count):
        build_up_end = float(np.exp(rng.normal(np.log(1.5), 0.3)))
        parameters = GroundMotionParameters(
            envelope_amplitude=float(np.exp(rng.normal(np.log(0.8), 0.6))),
            decay_rate=float(np.exp(rng.normal(np.log(0.5), 0.3))),
            decay_shape=float(np.exp(rng.normal(0.0, 0.15))),
            build_up_end_s=build_up_end,
            strong_phase_end_s=build_up_end + float(np.exp(rng.normal(np.log(3.0), 0.3))),
            start_frequency=float(np.exp(rng.normal(np.log(30.0), 0.2))),
            end_frequency=float(np.exp(rng.normal(np.log(12.0), 0.2))),
            filter_damping=float(np.clip(rng.normal(0.3, 0.08), 0.1, 0.6)),
        )
        records.append(FittedRecord(f"made-up-{index}", parameters, duration_s=20.0))
    return records
