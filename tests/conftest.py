"""Shared test inputs: plausible ground-motion parameters, made-up fitted records and a made-up PRJ-2048.

The made-up records stand in for real ESM fits so the pipeline can be tested without
downloading anything. They are not data and no result is drawn from them.
"""

import itertools

import numpy as np
import pytest

from earthquake_screening.buildings.dataset import SHAKING_INPUTS, SPECTRAL_PERIODS_S, Prj2048
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


def made_up_prj2048(motion_count: int = 20, seed: int = 0) -> Prj2048:
    """48 frames (1, 5 or 9 storeys) under made-up shaking, with drift a noisy power of Sa(1 s).

    A stand-in for PRJ-2048 so the building pipeline runs without the download.
    """
    rng = np.random.default_rng(seed)
    grid = itertools.product((1.0, 5.0, 9.0), (20.0, 30.0), (1.0, 1.5), (1.0, 3.0), (50.0, 80.0))
    designs = np.array(
        [[*design, (20.0, 67.5, 115.0)[index % 3]] for index, design in enumerate(grid)], dtype=np.float64
    )
    ln_pga = rng.normal(np.log(0.15), 0.6, motion_count)
    spectral = ln_pga[:, np.newaxis] + rng.normal(0.5, 0.3, (motion_count, len(SPECTRAL_PERIODS_S)))
    shaking = np.column_stack([ln_pga, spectral])
    ln_sa_1s = shaking[:, SHAKING_INPUTS.index("ln_sa_1s_g")]
    ln_drifts = -4.0 + 0.05 * designs[:, [0]] + ln_sa_1s[np.newaxis, :]
    noise = rng.normal(0.0, 0.2, ln_drifts.shape)
    return Prj2048(
        building_ids=np.arange(designs.shape[0], dtype=np.int64) * 2,
        designs=designs,
        shaking=shaking,
        peak_drifts=np.exp(ln_drifts + noise),
    )
