"""The ground-motion model: envelope shape, validation, unit-variance noise, high-pass, repeatability."""

import numpy as np
import pytest
from pydantic import ValidationError

from earthquake_screening.simulation.ground_motion import (
    GroundMotionParameters,
    envelope,
    high_pass,
    simulate_ground_motion,
    unit_variance_filtered_noise,
)


def test_envelope_rises_quadratically_then_holds_then_decays(typical_parameters: GroundMotionParameters) -> None:
    p = typical_parameters
    times = np.array([1.0, 2.0, 4.0, 6.0, 7.0])
    expected_after_decay = p.envelope_amplitude * np.exp(-p.decay_rate * 1.0**p.decay_shape)
    expected = [p.envelope_amplitude / 4, p.envelope_amplitude, p.envelope_amplitude, p.envelope_amplitude]
    assert envelope(times, p)[:4] == pytest.approx(expected)
    assert envelope(times, p)[4] == pytest.approx(expected_after_decay)


def test_strong_phase_ending_before_the_build_up_is_refused(typical_parameters: GroundMotionParameters) -> None:
    with pytest.raises(ValidationError, match="T2 must be after T1"):
        typical_parameters.replace(strong_phase_end_s=1.0)


def test_replace_validates_the_new_values(typical_parameters: GroundMotionParameters) -> None:
    with pytest.raises(ValidationError):
        typical_parameters.replace(filter_damping=1.5)


def test_parameters_round_trip_through_a_vector(typical_parameters: GroundMotionParameters) -> None:
    assert GroundMotionParameters.from_vector(typical_parameters.as_vector()) == typical_parameters


def test_filtered_noise_has_unit_variance_at_every_instant() -> None:
    rng = np.random.default_rng(0)
    sample_count, time_step = 1000, 0.01
    frequencies = np.linspace(30.0, 12.0, sample_count)
    realisations = np.array(
        [
            unit_variance_filtered_noise(rng.standard_normal(sample_count), time_step, frequencies, 0.3)
            for _ in range(400)
        ]
    )
    standard_deviation_over_realisations = realisations[:, [100, 500, 900]].std(axis=0)
    assert standard_deviation_over_realisations == pytest.approx([1.0, 1.0, 1.0], abs=0.1)


def test_high_pass_removes_a_constant_offset() -> None:
    corrected = high_pass(np.ones(6000), time_step=0.01)
    assert abs(corrected[-1]) < 1e-3


def test_same_seed_gives_the_same_record(typical_parameters: GroundMotionParameters) -> None:
    first = simulate_ground_motion(typical_parameters, 10.0, 0.01, np.random.default_rng(5))
    second = simulate_ground_motion(typical_parameters, 10.0, 0.01, np.random.default_rng(5))
    assert np.array_equal(first, second)


def test_infinite_parameters_are_refused(typical_parameters: GroundMotionParameters) -> None:
    with pytest.raises(ValidationError):
        typical_parameters.replace(envelope_amplitude=float("inf"))


def test_a_record_too_short_to_hold_its_phases_is_refused(typical_parameters: GroundMotionParameters) -> None:
    with pytest.raises(ValueError, match="too short"):
        simulate_ground_motion(typical_parameters, 0.05, 0.01, np.random.default_rng(0))
