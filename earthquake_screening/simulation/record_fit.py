"""Fit the eight ground-motion parameters to one real record (Sainct et al. 2020, section 2.2).

Three steps, as in the paper:
1. the envelope (alpha_1..3, T1, T2) and the delay T0 before the shaking starts: match
   the record's cumulative energy (Eq. 5-6). Real records start with some quiet
   pre-event noise; the paper fits T0 for real records and sets it to 0 when simulating,
   so steps 2 and 3 use the record from T0 on, and so does the synthetic length;
2. the filter's start and end frequencies: match the cumulative count of zero
   up-crossings (Eq. 8-10), a linear least-squares problem;
3. the filter damping: match the count of positive minima and negative maxima, by
   simulating the model at candidate dampings, since no formula gives that count.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize

from earthquake_screening.simulation.ground_motion import (
    MIN_SAMPLES,
    FloatArray,
    GroundMotionParameters,
    envelope,
    simulate_ground_motion,
)

# Candidate filter dampings for step 3, and how many simulations average out each count.
DAMPING_CANDIDATES = np.linspace(0.02, 0.9, 45)
SIMULATIONS_PER_DAMPING = 5
# Search limits for step 1, in the order amplitude (m/s^2), decay rate, decay shape,
# T1 (s), plateau length T2 - T1 (s), T0 (s). Wide enough for any real record; they
# stop the optimiser wandering to values that underflow (a plateau of 1e-16 s).
ENVELOPE_BOUNDS = ((1e-5, 1e3), (1e-4, 1e3), (0.05, 20.0), (1e-3, 600.0), (1e-3, 600.0), (1e-3, 600.0))
_LOG_ENVELOPE_BOUNDS = tuple((math.log(low), math.log(high)) for low, high in ENVELOPE_BOUNDS)
# Placeholder frequencies while fitting the envelope; step 1 doesn't depend on them.
_ANY_FREQUENCY = 10.0
_ANY_DAMPING = 0.3


class RecordFitError(ValueError):
    """The record can't be described by the model; it is counted and left out, never patched."""


def cumulative_energy(acceleration: FloatArray, time_step: float) -> FloatArray:
    return integrate.cumulative_trapezoid(acceleration**2, dx=time_step, initial=0)


@dataclass(frozen=True)
class EnvelopeFit:
    parameters: GroundMotionParameters  # the frequencies and damping are placeholders until steps 2 and 3
    start_delay_s: float  # T0


@dataclass(frozen=True)
class RecordFit:
    parameters: GroundMotionParameters
    start_delay_s: float


def _envelope_fit(log_values: FloatArray) -> EnvelopeFit:
    amplitude, decay_rate, decay_shape, build_up_end, plateau_length, start_delay = np.exp(log_values)
    parameters = GroundMotionParameters(
        envelope_amplitude=amplitude,
        decay_rate=decay_rate,
        decay_shape=decay_shape,
        build_up_end_s=build_up_end,
        strong_phase_end_s=build_up_end + plateau_length,
        start_frequency=_ANY_FREQUENCY,
        end_frequency=_ANY_FREQUENCY,
        filter_damping=_ANY_DAMPING,
    )
    return EnvelopeFit(parameters, float(start_delay))


def _initial_envelope_guess(energy: FloatArray, times: FloatArray) -> FloatArray:
    """Start from the record's 1%, 5%, 45% and 95% energy times, a usual description of its phases."""
    total = float(energy[-1])
    shortest = float(times[1])
    time_1, time_5, time_45, time_95 = (
        float(np.interp(fraction * total, energy, times)) for fraction in (0.01, 0.05, 0.45, 0.95)
    )
    start_delay = max(0.9 * time_1, shortest)
    build_up_end = max(time_5 - start_delay, shortest)
    plateau_length = max(time_45 - start_delay - build_up_end, shortest)
    amplitude = math.sqrt(0.9 * total / max(time_95 - time_5, shortest))
    decay_rate = 3.0 / max(time_95 - start_delay - build_up_end - plateau_length, shortest)
    return np.log(np.array([amplitude, decay_rate, 1.0, build_up_end, plateau_length, start_delay]))


def fit_envelope(acceleration: FloatArray, time_step: float) -> EnvelopeFit:
    """Step 1: least squares between the record's cumulative energy and the model's (Eq. 5-6)."""
    times = np.arange(acceleration.size, dtype=np.float64) * time_step
    record_energy = cumulative_energy(acceleration, time_step)
    total_energy = float(record_energy[-1])
    if total_energy <= 0:
        raise RecordFitError("the record is all zeros: there is nothing to fit")

    def mismatch(log_values: FloatArray) -> float:
        fit = _envelope_fit(log_values)
        model_energy = cumulative_energy(envelope(times - fit.start_delay_s, fit.parameters), time_step)
        return float(np.sum(((record_energy - model_energy) / total_energy) ** 2))

    lower, upper = np.array(_LOG_ENVELOPE_BOUNDS).T
    start = np.clip(_initial_envelope_guess(record_energy, times), lower, upper)
    result = optimize.minimize(
        mismatch, start, method="Nelder-Mead", bounds=_LOG_ENVELOPE_BOUNDS, options={"maxiter": 6000}
    )
    return _envelope_fit(np.asarray(result.x, dtype=np.float64))


def zero_up_crossing_times(acceleration: FloatArray, time_step: float) -> FloatArray:
    crossing = (acceleration[:-1] < 0) & (acceleration[1:] >= 0)
    return (np.flatnonzero(crossing) + 1).astype(np.float64) * time_step


def fit_filter_frequencies(acceleration: FloatArray, time_step: float) -> tuple[float, float]:
    """Step 2: the up-crossing count grows like (w0*t + (wn - w0)*t^2 / (2T)) / (2*pi) (Eq. 8-10)."""
    duration = acceleration.size * time_step
    crossing_times = zero_up_crossing_times(acceleration, time_step)
    if crossing_times.size < 3:
        raise RecordFitError(f"only {crossing_times.size} zero up-crossings: too few to fit the filter frequencies")
    counts = np.arange(1, crossing_times.size + 1, dtype=np.float64)
    design = np.column_stack([crossing_times, crossing_times**2])
    (linear, quadratic), *_ = np.linalg.lstsq(design, counts, rcond=None)
    start_frequency = 2 * math.pi * float(linear)
    end_frequency = start_frequency + 4 * math.pi * duration * float(quadratic)
    if start_frequency <= 0 or end_frequency <= 0:
        raise RecordFitError(
            f"fitted filter frequencies {start_frequency:.2f} and {end_frequency:.2f} rad/s: both must be positive"
        )
    return start_frequency, end_frequency


def count_positive_minima_and_negative_maxima(acceleration: FloatArray) -> int:
    middle = acceleration[1:-1]
    is_minimum = (middle < acceleration[:-2]) & (middle < acceleration[2:])
    is_maximum = (middle > acceleration[:-2]) & (middle > acceleration[2:])
    return int(np.sum(is_minimum & (middle > 0)) + np.sum(is_maximum & (middle < 0)))


def fit_filter_damping(
    acceleration: FloatArray, time_step: float, partial_fit: GroundMotionParameters, seed: int
) -> float:
    """Step 3: the candidate damping whose simulations best match the record's count."""
    target = count_positive_minima_and_negative_maxima(acceleration)
    duration = acceleration.size * time_step
    mismatches: list[float] = []
    for damping in DAMPING_CANDIDATES:
        candidate = partial_fit.replace(filter_damping=float(damping))
        # The same seeds for every candidate, so the comparison isn't blurred by noise.
        rng = np.random.default_rng(seed)
        counts = [
            count_positive_minima_and_negative_maxima(simulate_ground_motion(candidate, duration, time_step, rng))
            for _ in range(SIMULATIONS_PER_DAMPING)
        ]
        mismatches.append(abs(float(np.mean(counts)) - target))
    return float(DAMPING_CANDIDATES[int(np.argmin(mismatches))])


def fit_record(acceleration: FloatArray, time_step: float, seed: int) -> RecordFit:
    """All eight parameters for one record, in m/s^2 and seconds, and the delay T0 to drop."""
    envelope_fit = fit_envelope(acceleration, time_step)
    shaking = acceleration[round(envelope_fit.start_delay_s / time_step) :]
    if shaking.size < MIN_SAMPLES:
        raise RecordFitError(f"only {shaking.size} samples after the fitted start delay: too few to fit the filter")
    start_frequency, end_frequency = fit_filter_frequencies(shaking, time_step)
    partial_fit = envelope_fit.parameters.replace(start_frequency=start_frequency, end_frequency=end_frequency)
    damping = fit_filter_damping(shaking, time_step, partial_fit, seed)
    return RecordFit(partial_fit.replace(filter_damping=damping), envelope_fit.start_delay_s)
