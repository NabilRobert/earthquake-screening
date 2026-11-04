"""The pool: the window on L, labels, failed runs kept and counted, and saving."""

from pathlib import Path

import numpy as np

from earthquake_screening.simulation.ground_motion import GroundMotionParameters
from earthquake_screening.simulation.oscillator import Oscillator
from earthquake_screening.simulation.pool import Pool, build_pool, simulate_record
from earthquake_screening.simulation.synthesis import SyntheticRecordSpec


def test_a_record_too_weak_for_the_window_is_not_run_nonlinearly(typical_parameters: GroundMotionParameters) -> None:
    weak = SyntheticRecordSpec(typical_parameters.replace(envelope_amplitude=1e-4), 10.0, "weak", seed=1)
    simulated = simulate_record(weak, Oscillator())
    assert not simulated.is_in_window
    assert np.isnan(simulated.nonlinear_peak)


def test_a_record_in_the_window_gets_a_nonlinear_peak(typical_parameters: GroundMotionParameters) -> None:
    # Amplitude chosen so L lands between Y and 6Y for this seed.
    moderate = SyntheticRecordSpec(typical_parameters.replace(envelope_amplitude=1.5), 10.0, "moderate", seed=1)
    simulated = simulate_record(moderate, Oscillator())
    assert simulated.is_in_window
    assert simulated.nonlinear_peak > 0


def _pool(nonlinear_peak: list[float], did_converge: list[bool]) -> Pool:
    count = len(nonlinear_peak)
    return Pool(
        parameters=np.ones((count, 8)),
        intensity=np.ones((count, 5)),
        is_in_window=np.ones(count, dtype=bool),
        nonlinear_peak=np.array(nonlinear_peak),
        did_converge=np.array(did_converge),
        failure_displacement_m=0.01,
    )


def test_failure_means_a_peak_over_twice_the_yield() -> None:
    pool = _pool([0.009, 0.011], [True, True])
    assert pool.has_failed.tolist() == [False, True]


def test_runs_that_did_not_converge_are_kept_but_not_labelled() -> None:
    pool = _pool([0.009, float("nan")], [True, False])
    assert pool.labelled_rows.tolist() == [True, False]
    assert pool.nonlinear_peak.size == 2


def test_pool_round_trips_through_a_file(tmp_path: Path) -> None:
    pool = _pool([0.009, 0.011], [True, True])
    path = tmp_path / "pool.npz"
    pool.save(str(path))
    loaded = Pool.load(str(path))
    assert np.array_equal(loaded.nonlinear_peak, pool.nonlinear_peak)
    assert loaded.failure_displacement_m == pool.failure_displacement_m


def test_the_same_specs_give_the_same_pool_across_processes(typical_parameters: GroundMotionParameters) -> None:
    specs = [
        SyntheticRecordSpec(typical_parameters.replace(envelope_amplitude=1.5), 10.0, f"r{seed}", seed=seed)
        for seed in range(8)
    ]
    first = build_pool(specs, Oscillator(), workers=2)
    second = build_pool(specs, Oscillator(), workers=3)
    assert np.array_equal(first.intensity, second.intensity)
    assert np.array_equal(first.nonlinear_peak, second.nonlinear_peak, equal_nan=True)
