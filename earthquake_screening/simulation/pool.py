"""Simulate every synthetic record and label it: the pool the active learner draws from.

For each record (Sainct et al. 2020, sections 3.1 and 4.3):
1. simulate the ground motion and its intensity measures, including L, the peak of
   the linear oscillator;
2. keep it only if Y <= L <= 6Y, as the paper does: below, the oscillator stays
   elastic and can't fail; above, it almost surely fails;
3. run the nonlinear oscillator in OpenSees; failure means a peak over 2Y.

Records run in parallel processes; each has its own seed, so the pool is the same on any machine.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

from earthquake_screening.simulation.ground_motion import FloatArray, simulate_ground_motion
from earthquake_screening.simulation.oscillator import Oscillator, linear_peak_displacement, nonlinear_peak_displacement
from earthquake_screening.simulation.synthesis import SyntheticRecordSpec

# The paper gives no time step for synthetic records. 0.01 s resolves the filter's
# frequencies (Nyquist 50 Hz) and the 5 Hz oscillator with 20 samples a cycle.
SYNTHETIC_TIME_STEP_S = 0.01
# The window on L, in multiples of the yield displacement (section 4.3).
LOWEST_KEPT_PEAK_IN_YIELDS = 1.0
HIGHEST_KEPT_PEAK_IN_YIELDS = 6.0

BoolArray = NDArray[np.bool_]

INTENSITY_NAMES = ("pga", "pgv", "pgd", "energy", "linear_peak")


@dataclass(frozen=True)
class SimulatedRecord:
    intensity: tuple[float, float, float, float, float]  # in the order of INTENSITY_NAMES, SI units
    is_in_window: bool
    nonlinear_peak: float  # NaN when out of the window or when OpenSees didn't converge
    did_converge: bool


def intensity_measures(acceleration: FloatArray, time_step: float, oscillator: Oscillator) -> tuple[float, ...]:
    """PGA, PGV, PGD, total energy (integral of a^2) and L: the paper's record features (section 3.3)."""
    velocity = integrate.cumulative_trapezoid(acceleration, dx=time_step, initial=0)
    displacement = integrate.cumulative_trapezoid(velocity, dx=time_step, initial=0)
    return (
        float(np.max(np.abs(acceleration))),
        float(np.max(np.abs(velocity))),
        float(np.max(np.abs(displacement))),
        float(integrate.trapezoid(acceleration**2, dx=time_step)),
        linear_peak_displacement(acceleration, time_step, oscillator),
    )


def simulate_record(spec: SyntheticRecordSpec, oscillator: Oscillator) -> SimulatedRecord:
    rng = np.random.default_rng(spec.seed)
    acceleration = simulate_ground_motion(spec.parameters, spec.duration_s, SYNTHETIC_TIME_STEP_S, rng)
    pga, pgv, pgd, energy, linear_peak = intensity_measures(acceleration, SYNTHETIC_TIME_STEP_S, oscillator)
    intensity = (pga, pgv, pgd, energy, linear_peak)
    yield_displacement = oscillator.yield_displacement_m
    is_in_window = (
        LOWEST_KEPT_PEAK_IN_YIELDS * yield_displacement
        <= linear_peak
        <= HIGHEST_KEPT_PEAK_IN_YIELDS * yield_displacement
    )
    if not is_in_window:
        return SimulatedRecord(intensity, is_in_window=False, nonlinear_peak=float("nan"), did_converge=True)
    try:
        peak = nonlinear_peak_displacement(acceleration, SYNTHETIC_TIME_STEP_S, oscillator)
    except RuntimeError:
        # Kept and counted, never dropped quietly (requirement S3).
        return SimulatedRecord(intensity, is_in_window=True, nonlinear_peak=float("nan"), did_converge=False)
    return SimulatedRecord(intensity, is_in_window=True, nonlinear_peak=peak, did_converge=True)


@dataclass(frozen=True)
class Pool:
    """One row per synthetic record; the labelled pool is the rows in the window that converged."""

    parameters: FloatArray  # (records, 8), ground-motion parameters in PARAMETER_NAMES order
    intensity: FloatArray  # (records, 5), in INTENSITY_NAMES order
    is_in_window: BoolArray
    nonlinear_peak: FloatArray
    did_converge: BoolArray
    failure_displacement_m: float

    @property
    def labelled_rows(self) -> BoolArray:
        return self.is_in_window & self.did_converge

    @property
    def has_failed(self) -> BoolArray:
        """Failure: the nonlinear peak exceeds twice the yield displacement (section 4.2)."""
        return np.nan_to_num(self.nonlinear_peak, nan=0.0) > self.failure_displacement_m

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            parameters=self.parameters,
            intensity=self.intensity,
            is_in_window=self.is_in_window,
            nonlinear_peak=self.nonlinear_peak,
            did_converge=self.did_converge,
            failure_displacement_m=np.array(self.failure_displacement_m),
        )

    @classmethod
    def load(cls, path: str) -> "Pool":
        with np.load(path) as stored:
            return cls(
                parameters=stored["parameters"],
                intensity=stored["intensity"],
                is_in_window=stored["is_in_window"],
                nonlinear_peak=stored["nonlinear_peak"],
                did_converge=stored["did_converge"],
                failure_displacement_m=float(stored["failure_displacement_m"]),
            )


def build_pool(specs: list[SyntheticRecordSpec], oscillator: Oscillator, workers: int | None = None) -> Pool:
    """Simulate all records, on every core by default."""
    # OpenSees holds one model per process, so the parallelism is processes, not threads.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        simulated = list(executor.map(partial(simulate_record, oscillator=oscillator), specs, chunksize=64))
    return Pool(
        parameters=np.array([spec.parameters.as_vector() for spec in specs]),
        intensity=np.array([record.intensity for record in simulated]),
        is_in_window=np.array([record.is_in_window for record in simulated]),
        nonlinear_peak=np.array([record.nonlinear_peak for record in simulated]),
        did_converge=np.array([record.did_converge for record in simulated]),
        failure_displacement_m=oscillator.failure_displacement_m,
    )
