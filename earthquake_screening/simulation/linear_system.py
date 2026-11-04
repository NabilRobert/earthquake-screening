"""Exact response of a small linear system to a sampled record, shared by the oscillator and the high-pass filter.

The system is discretised with a first-order hold, which is exact for a record that
varies linearly between samples, then run as a digital filter. scipy's dlsim gives the
same numbers but steps through the record in Python, about 50 times slower.
"""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy import signal

FloatArray = NDArray[np.float64]
# (state, input, output, feedthrough) matrices as nested tuples, so they can be cache keys.
Matrices = tuple[tuple[tuple[float, ...], ...], ...]


@lru_cache(maxsize=16)
def _digital_filter(matrices: Matrices, time_step: float) -> tuple[FloatArray, FloatArray]:
    state, inputs, outputs, feedthrough = (np.array(matrix, dtype=np.float64) for matrix in matrices)
    discrete = signal.cont2discrete((state, inputs, outputs, feedthrough), time_step, method="foh")
    numerator, denominator = signal.ss2tf(*discrete[:4])
    return np.asarray(numerator[0], dtype=np.float64), np.asarray(denominator, dtype=np.float64)


def respond(matrices: Matrices, time_step: float, record: FloatArray) -> FloatArray:
    """The system's single output over time, starting from rest."""
    numerator, denominator = _digital_filter(matrices, time_step)
    return np.asarray(signal.lfilter(numerator, denominator, record), dtype=np.float64)


def peak_oscillator_displacement(
    ground_acceleration: FloatArray, time_step: float, angular_frequency: float, damping_ratio: float
) -> float:
    """Peak |relative displacement| of a linear single-mass oscillator under a ground acceleration.

    z'' + 2*zeta*w*z' + w^2*z = -a_g. Used for slice 1's intensity measure L and for the
    response spectra of slice 2's ground motions.
    """
    matrices = (
        ((0.0, 1.0), (-(angular_frequency**2), -2 * damping_ratio * angular_frequency)),
        ((0.0,), (-1.0,)),
        ((1.0, 0.0),),
        ((0.0,),),
    )
    return float(np.max(np.abs(respond(matrices, time_step, ground_acceleration))))
