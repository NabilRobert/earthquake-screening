"""The single-mass oscillator of Sainct et al. (2020), section 4.2.

Two versions of the same oscillator under a ground acceleration:
- linear, whose peak displacement is the intensity measure L (their Eq. 20),
  solved exactly with scipy because it runs on all 100,000 synthetic records;
- nonlinear (bilinear, kinematic hardening), whose peak displacement decides
  failure, solved with OpenSeesPy because that's the field's standard engine.

OpenSeesPy is imported here only. It is free for research and internal use;
nothing that serves results may import it (licence gate, ADR-002).
"""

import math
from dataclasses import dataclass

import numpy as np
import openseespy.opensees as ops
from numpy.typing import NDArray

from earthquake_screening.simulation.linear_system import peak_oscillator_displacement

FloatArray = NDArray[np.float64]

# Newmark's average-acceleration step distorts broadband input: at the records' 0.01 s
# step the peak was 6.5% off a 40-substep reference in spike 3. Five substeps per
# sample bring it under 0.4% for about 24 ms a record (docs/spikes.md).
SUBSTEPS_PER_SAMPLE = 5


@dataclass(frozen=True)
class Oscillator:
    """The paper's base case: 5 Hz, 2% damping, yield at 5 mm, 20% post-yield stiffness."""

    frequency_hz: float = 5.0
    damping_ratio: float = 0.02
    yield_displacement_m: float = 5e-3
    post_yield_stiffness_ratio: float = 0.2

    def __post_init__(self) -> None:
        values = (self.frequency_hz, self.damping_ratio, self.yield_displacement_m, self.post_yield_stiffness_ratio)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError(f"oscillator values must be finite and positive, got {self}")

    @property
    def angular_frequency(self) -> float:
        return 2 * math.pi * self.frequency_hz

    @property
    def failure_displacement_m(self) -> float:
        """Failure is a peak displacement over twice the yield displacement (section 4.2)."""
        return 2 * self.yield_displacement_m


def linear_peak_displacement(ground_acceleration: FloatArray, time_step: float, oscillator: Oscillator) -> float:
    """Peak |relative displacement| of the linear oscillator: the intensity measure L (Eq. 20).

    z'' + 2*zeta*w*z' + w^2*z = -a_g, solved exactly for a piecewise-linear record: within
    0.8% of a 20-substep reference in spike 3, where Newmark at the same step was 6% off.
    """
    return peak_oscillator_displacement(
        ground_acceleration, time_step, oscillator.angular_frequency, oscillator.damping_ratio
    )


def _build_opensees_model(ground_acceleration: FloatArray, time_step: float, oscillator: Oscillator) -> None:
    """Unit mass, so the stiffness is w^2 and the damping 2*zeta*w (mass-proportional,
    which is exactly the paper's linear viscous term for one degree of freedom).
    Steel01 without its isotropic-hardening options is bilinear with kinematic hardening.
    """
    stiffness = oscillator.angular_frequency**2
    yield_force = stiffness * oscillator.yield_displacement_m
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)
    ops.node(2, 0.0, "-mass", 1.0)
    ops.fix(1, 1)
    ops.uniaxialMaterial("Steel01", 1, yield_force, stiffness, oscillator.post_yield_stiffness_ratio)
    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)
    ops.rayleigh(2 * oscillator.damping_ratio * oscillator.angular_frequency, 0.0, 0.0, 0.0)
    ops.timeSeries("Path", 1, "-dt", time_step, "-values", *ground_acceleration.tolist())
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("ProfileSPD")
    ops.test("NormDispIncr", 1e-10, 50)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")


def nonlinear_peak_displacement(ground_acceleration: FloatArray, time_step: float, oscillator: Oscillator) -> float:
    """Peak |relative displacement| of the bilinear oscillator, by OpenSees time-history analysis."""
    if not np.all(np.isfinite(ground_acceleration)):
        raise ValueError("the ground acceleration contains NaN or infinity: check the record before simulating")
    _build_opensees_model(ground_acceleration, time_step, oscillator)
    # The Path series interpolates linearly between samples, so substeps need no resampling.
    substep = time_step / SUBSTEPS_PER_SAMPLE
    peak = 0.0
    for _ in range(len(ground_acceleration) * SUBSTEPS_PER_SAMPLE):
        if ops.analyze(1, substep) != 0:
            raise RuntimeError(
                "OpenSees did not converge on this record: keep it and count it as a failed run (requirement S3)"
            )
        peak = max(peak, abs(ops.nodeDisp(2, 1)))
    ops.wipe()
    if not math.isfinite(peak):
        # A NaN peak would compare as "didn't fail" and mislabel the record.
        raise RuntimeError("OpenSees returned a non-finite displacement: count it as a failed run (requirement S3)")
    return peak
