"""The oscillator: exact linear peaks, OpenSees agreeing while elastic, and the bilinear law."""

import numpy as np
import pytest

from earthquake_screening.simulation.oscillator import (
    Oscillator,
    linear_peak_displacement,
    nonlinear_peak_displacement,
)

TIME_STEP = 0.01


def test_linear_peak_under_slow_shaking_is_the_static_displacement() -> None:
    # A 0.05 Hz sine is far below 5 Hz, so z ~ a / w^2, amplified by 1 / (1 - (f / 5)^2) ~ 1.0001.
    oscillator = Oscillator()
    times = np.arange(0.0, 40.0, TIME_STEP)
    amplitude = 2.0
    acceleration = amplitude * np.sin(2 * np.pi * 0.05 * times)
    expected = amplitude / oscillator.angular_frequency**2
    assert linear_peak_displacement(acceleration, TIME_STEP, oscillator) == pytest.approx(expected, rel=0.01)


def test_opensees_agrees_with_the_exact_linear_solution_while_elastic() -> None:
    elastic = Oscillator(yield_displacement_m=10.0)
    rng = np.random.default_rng(1)
    times = np.arange(0.0, 20.0, TIME_STEP)
    acceleration = rng.standard_normal(times.size) * np.exp(-(((times - 6) / 3) ** 2))
    linear = linear_peak_displacement(acceleration, TIME_STEP, elastic)
    assert nonlinear_peak_displacement(acceleration, TIME_STEP, elastic) == pytest.approx(linear, rel=0.01)


def test_slow_push_past_yield_follows_the_bilinear_law() -> None:
    # Quasi-static push to a force of 2*k*Y: the yield takes k*Y, the 20% post-yield
    # stiffness takes the rest over (2*k*Y - k*Y) / (0.2*k) = 5*Y, so the peak is 6*Y.
    oscillator = Oscillator()
    stiffness = oscillator.angular_frequency**2
    times = np.arange(0.0, 60.0, TIME_STEP)
    acceleration = -2 * stiffness * oscillator.yield_displacement_m * np.minimum(times / 50.0, 1.0)
    peak = nonlinear_peak_displacement(acceleration, TIME_STEP, oscillator)
    assert peak == pytest.approx(6 * oscillator.yield_displacement_m, rel=0.02)


def test_failure_is_twice_the_yield_displacement() -> None:
    assert Oscillator().failure_displacement_m == pytest.approx(0.01)


def test_an_oscillator_with_zero_yield_is_refused() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        Oscillator(yield_displacement_m=0.0)


def test_a_record_with_nan_is_refused_before_opensees_runs() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        nonlinear_peak_displacement(np.array([0.0, float("nan"), 0.0]), TIME_STEP, Oscillator())
