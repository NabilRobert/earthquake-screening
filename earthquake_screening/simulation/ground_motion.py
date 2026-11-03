"""Synthetic ground motions: the Rezaeian and Der Kiureghian model as Sainct et al. (2020) use it.

A record is white noise passed through a filter whose frequency drifts over time,
scaled to unit variance, shaped by a time envelope q(t), then high-pass corrected:
- envelope q(t): Eq. 3, a quadratic build-up to T1, a plateau to T2, then a decay;
- filter: Eq. 2, a damped oscillator whose frequency goes from w0 to wn over the record;
- correction: Eq. 4, a 0.2 Hz high-pass oscillator, so velocity and displacement don't drift.

Eight parameters describe a record (the paper says 9 in one place and 8 in another;
8 is the list it actually uses, docs/reproduction/sainct2020.md).
"""

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from earthquake_screening.simulation.linear_system import respond

FloatArray = NDArray[np.float64]

HIGH_PASS_CORNER_HZ = 0.2  # Eq. 4
# Lags beyond the point where every filter impulse has decayed to this fraction add
# nothing measurable, and skipping them is what makes 100,000 records affordable.
IMPULSE_DECAY_CUTOFF = 1e-6
# Fewer samples than this can't hold a build-up, a strong phase and a decay.
MIN_SAMPLES = 10


class GroundMotionParameters(BaseModel):
    """The eight parameters of one record (paper symbols in brackets). Units: m/s^2, s, rad/s."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    envelope_amplitude: float = Field(gt=0)  # alpha_1
    decay_rate: float = Field(gt=0)  # alpha_2
    decay_shape: float = Field(gt=0)  # alpha_3
    build_up_end_s: float = Field(gt=0)  # T_1
    strong_phase_end_s: float = Field(gt=0)  # T_2
    start_frequency: float = Field(gt=0)  # omega_0
    end_frequency: float = Field(gt=0)  # omega_n
    filter_damping: float = Field(gt=0, lt=1)  # zeta_f

    @model_validator(mode="after")
    def _phases_in_order(self) -> "GroundMotionParameters":
        if self.strong_phase_end_s <= self.build_up_end_s:
            raise ValueError(
                f"strong phase ends at {self.strong_phase_end_s} s, before the build-up ends at "
                f"{self.build_up_end_s} s: T2 must be after T1"
            )
        return self

    def replace(self, **changes: float) -> "GroundMotionParameters":
        """A copy with some values changed, validated again (pydantic's model_copy skips validation)."""
        return GroundMotionParameters.model_validate({**self.model_dump(), **changes})

    def as_vector(self) -> FloatArray:
        return np.array([getattr(self, name) for name in PARAMETER_NAMES], dtype=np.float64)

    @classmethod
    def from_vector(cls, vector: FloatArray) -> "GroundMotionParameters":
        return cls(**dict(zip(PARAMETER_NAMES, (float(value) for value in vector), strict=True)))


PARAMETER_NAMES: tuple[str, ...] = tuple(GroundMotionParameters.model_fields)


def envelope(times: FloatArray, parameters: GroundMotionParameters) -> FloatArray:
    """The time envelope q(t) of Eq. 3, with T0 = 0 as in the paper's simulations."""
    build_up = parameters.envelope_amplitude * (times / parameters.build_up_end_s) ** 2
    time_after_peak = np.clip(times - parameters.strong_phase_end_s, 0.0, None)
    decay = parameters.envelope_amplitude * np.exp(-parameters.decay_rate * time_after_peak**parameters.decay_shape)
    plateau = np.full_like(times, parameters.envelope_amplitude)
    return np.where(
        times <= parameters.build_up_end_s,
        build_up,
        np.where(times <= parameters.strong_phase_end_s, plateau, decay),
    )


def filter_frequencies(times: FloatArray, duration_s: float, parameters: GroundMotionParameters) -> FloatArray:
    """The filter's frequency at each excitation time: linear from w0 to wn over the record."""
    return parameters.start_frequency + (times / duration_s) * (parameters.end_frequency - parameters.start_frequency)


def unit_variance_filtered_noise(
    white_noise: FloatArray, time_step: float, frequencies: FloatArray, damping: float
) -> FloatArray:
    """Filter white noise with a time-varying oscillator and scale it to unit variance at every instant.

    Each impulse rings at the frequency of the moment it arrived (Eq. 2), so the
    response isn't a convolution and an FFT can't do it. We add the ring-down of all
    impulses lag by lag instead, and stop once even the slowest has died out.
    """
    sample_count = white_noise.size
    damped_factor = math.sqrt(1 - damping**2)
    slowest_decay = damping * float(np.min(frequencies))
    max_lag = min(
        sample_count,
        math.ceil(-math.log(IMPULSE_DECAY_CUTOFF) / (slowest_decay * time_step)) + 1,
    )
    filtered = np.zeros(sample_count)
    variance = np.zeros(sample_count)
    for lag in range(1, max_lag):
        elapsed = lag * time_step
        arrival_frequencies = frequencies[: sample_count - lag]
        impulse = (
            arrival_frequencies
            / damped_factor
            * np.exp(-damping * arrival_frequencies * elapsed)
            * np.sin(arrival_frequencies * damped_factor * elapsed)
        )
        filtered[lag:] += impulse * white_noise[: sample_count - lag]
        variance[lag:] += impulse**2
    standard_deviation = np.sqrt(variance)
    return np.divide(
        filtered,
        standard_deviation,
        out=np.zeros(sample_count),
        where=standard_deviation > 0,
    )


def high_pass(acceleration: FloatArray, time_step: float) -> FloatArray:
    """Eq. 4: u'' + 2*wc*u' + wc^2*u = s, and the corrected record is u'' = s - 2*wc*u' - wc^2*u."""
    corner = 2 * math.pi * HIGH_PASS_CORNER_HZ
    matrices = (
        ((0.0, 1.0), (-(corner**2), -2 * corner)),
        ((0.0,), (1.0,)),
        ((-(corner**2), -2 * corner),),
        ((1.0,),),
    )
    return respond(matrices, time_step, acceleration)


def simulate_ground_motion(
    parameters: GroundMotionParameters,
    duration_s: float,
    time_step: float,
    rng: np.random.Generator,
) -> FloatArray:
    """One synthetic ground acceleration record (m/s^2), sampled every time_step seconds."""
    if not duration_s >= MIN_SAMPLES * time_step > 0:
        raise ValueError(
            f"a {duration_s} s record at a {time_step} s step is too short: need at least {MIN_SAMPLES} samples"
        )
    times = np.arange(0.0, duration_s, time_step)
    white_noise = rng.standard_normal(times.size)
    frequencies = filter_frequencies(times, duration_s, parameters)
    unit_noise = unit_variance_filtered_noise(white_noise, time_step, frequencies, parameters.filter_damping)
    return high_pass(envelope(times, parameters) * unit_noise, time_step)
