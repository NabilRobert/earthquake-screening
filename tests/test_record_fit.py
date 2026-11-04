"""Fitting a record: each step recovers what a known synthetic record was made from."""

import numpy as np
import pytest

from earthquake_screening.simulation.ground_motion import GroundMotionParameters, envelope, simulate_ground_motion
from earthquake_screening.simulation.record_fit import (
    RecordFitError,
    count_positive_minima_and_negative_maxima,
    cumulative_energy,
    fit_envelope,
    fit_filter_damping,
    fit_filter_frequencies,
    fit_record,
    zero_up_crossing_times,
)

TIME_STEP = 0.005
DURATION = 30.0


@pytest.fixture
def known_record(typical_parameters: GroundMotionParameters) -> np.ndarray:
    return simulate_ground_motion(typical_parameters, DURATION, TIME_STEP, np.random.default_rng(0))


def _energy_gap(record: np.ndarray, parameters: GroundMotionParameters, start_delay: float) -> float:
    times = np.arange(record.size, dtype=np.float64) * TIME_STEP
    model_energy = cumulative_energy(envelope(times - start_delay, parameters), TIME_STEP)
    return float(np.sum((cumulative_energy(record, TIME_STEP) - model_energy) ** 2))


def test_fitted_envelope_matches_the_record_energy_at_least_as_well_as_the_truth(
    known_record: np.ndarray, typical_parameters: GroundMotionParameters
) -> None:
    # One realisation's energy curve wanders 10-20% from the envelope that made it, so
    # the fit can't be held to the true parameters; it must match the record no worse.
    fit = fit_envelope(known_record, TIME_STEP)
    assert _energy_gap(known_record, fit.parameters, fit.start_delay_s) <= _energy_gap(
        known_record, typical_parameters, 0.0
    )


def test_a_quiet_lead_in_moves_the_fitted_start_by_its_length(known_record: np.ndarray) -> None:
    # T0 also takes in the quiet start of the quadratic build-up, so it isn't 0 without a
    # lead-in; what must hold is that 5 s of pre-event noise moves it by about 5 s.
    lead_in = 1e-4 * np.random.default_rng(3).standard_normal(round(5.0 / TIME_STEP))
    without = fit_envelope(known_record, TIME_STEP).start_delay_s
    with_lead_in = fit_envelope(np.concatenate([lead_in, known_record]), TIME_STEP).start_delay_s
    assert with_lead_in - without == pytest.approx(5.0, abs=1.0)


def test_fit_recovers_the_filter_frequencies(
    known_record: np.ndarray, typical_parameters: GroundMotionParameters
) -> None:
    start, end = fit_filter_frequencies(known_record, TIME_STEP)
    assert start == pytest.approx(typical_parameters.start_frequency, rel=0.15)
    assert end == pytest.approx(typical_parameters.end_frequency, rel=0.15)


def test_fit_recovers_the_filter_damping(known_record: np.ndarray, typical_parameters: GroundMotionParameters) -> None:
    start, end = fit_filter_frequencies(known_record, TIME_STEP)
    partial = typical_parameters.replace(start_frequency=start, end_frequency=end)
    assert fit_filter_damping(known_record, TIME_STEP, partial, seed=1) == pytest.approx(0.3, abs=0.1)


def test_an_all_zero_record_is_refused() -> None:
    with pytest.raises(RecordFitError, match="all zeros"):
        fit_envelope(np.zeros(1000), TIME_STEP)


def test_a_record_with_too_few_crossings_is_refused() -> None:
    with pytest.raises(RecordFitError, match="too few"):
        fit_filter_frequencies(np.array([-1.0, 1.0, 1.0, 1.0]), TIME_STEP)


def test_zero_up_crossings_are_found_where_the_sign_turns_positive() -> None:
    record = np.array([-1.0, 1.0, -1.0, -1.0, 0.0, 2.0])
    assert zero_up_crossing_times(record, 0.5) == pytest.approx([0.5, 2.0])


def test_only_positive_minima_and_negative_maxima_are_counted() -> None:
    # A positive minimum at 2, a negative maximum at -1, a negative minimum and a positive maximum ignored.
    record = np.array([5.0, 2.0, 6.0, -3.0, -1.0, -4.0, 0.0, 7.0, 1.0])
    assert count_positive_minima_and_negative_maxima(record) == 2


def test_the_frequencies_are_fitted_on_the_shaking_after_the_lead_in(
    known_record: np.ndarray, typical_parameters: GroundMotionParameters
) -> None:
    # Pre-event noise crosses zero at nearly every sample; left in, it would drag the
    # start frequency far above the record's.
    lead_in = 1e-4 * np.random.default_rng(3).standard_normal(round(5.0 / TIME_STEP))
    fit = fit_record(np.concatenate([lead_in, known_record]), TIME_STEP, seed=1)
    assert fit.parameters.start_frequency == pytest.approx(typical_parameters.start_frequency, rel=0.2)
