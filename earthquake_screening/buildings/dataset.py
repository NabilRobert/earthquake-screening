"""Read DesignSafe PRJ-2048 into arrays: each frame's design, each ground motion's shaking, and the peak drifts.

The files come from data/prj2048/ (`earthquake-screening fetch-prj2048`). Each frame's
drift file has one row per storey and one column per ground motion (GM1 to GM240);
the peak inter-storey drift is the largest over storeys. Ground motion k is
Histories/k.txt, one acceleration per line, in g (the files don't say; their peaks,
0.03 to 0.78, only make sense in g). The shaking is described by its response
spectrum, because that is what a hazard tool gives a user.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from earthquake_screening.simulation.linear_system import peak_oscillator_displacement

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# DesignSummary.csv's columns, in the order of BUILDING_INPUTS. The column-beam ratio
# is 1.0 for every frame, so it carries no information and is left out.
DESIGN_COLUMNS = (
    "number of story",
    "bay width",
    "story height ratio",
    "number of bay",
    "floor dead load",
    "roof dead load",
)
BUILDING_INPUTS = (
    "storeys",
    "bay_width_ft",
    "story_height_ratio",
    "bays",
    "floor_dead_load_psf",
    "roof_dead_load_psf",
)
SPECTRAL_PERIODS_S = (0.2, 0.5, 1.0, 2.0, 4.0)
SPECTRUM_DAMPING_RATIO = 0.05  # the convention for design and hazard spectra
SHAKING_INPUTS = ("ln_pga_g", *(f"ln_sa_{period:g}s_g" for period in SPECTRAL_PERIODS_S))
INPUT_NAMES = BUILDING_INPUTS + SHAKING_INPUTS
STOREYS_COLUMN = BUILDING_INPUTS.index("storeys")
GROUND_MOTION_COUNT = 240

DRIFT_FOLDER = Path("StructuralResponses/EDPsUnder240GMs/PeakStoryDrift")
MOTION_FOLDER = Path("GroundMotions/MirandaGroundMotions")


@dataclass(frozen=True)
class Prj2048:
    building_ids: IntArray  # PRJ-2048's own IDs, which skip numbers
    designs: FloatArray  # one row per frame, columns BUILDING_INPUTS
    shaking: FloatArray  # one row per ground motion, columns SHAKING_INPUTS
    peak_drifts: FloatArray  # frames x ground motions: the largest storey drift ratio


def read_designs(path: Path) -> tuple[IntArray, FloatArray]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    building_ids = np.array([int(float(row["building ID"])) for row in rows], dtype=np.int64)
    designs = np.array([[float(row[column]) for column in DESIGN_COLUMNS] for row in rows], dtype=np.float64)
    return building_ids, designs


def read_peak_drifts(folder: Path, building_ids: IntArray) -> FloatArray:
    """The largest storey drift ratio of each frame under each ground motion."""
    peaks = np.empty((building_ids.size, GROUND_MOTION_COUNT), dtype=np.float64)
    for row, building_id in enumerate(building_ids):
        path = folder / f"Building_{building_id}_PSDR.csv"
        storey_drifts = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if storey_drifts.shape[1] != GROUND_MOTION_COUNT:
            raise ValueError(f"{path}: {storey_drifts.shape[1]} ground motions, expected {GROUND_MOTION_COUNT}")
        peaks[row] = storey_drifts.max(axis=0)
    if not np.all(np.isfinite(peaks) & (peaks > 0)):
        raise ValueError("a peak drift is missing, zero or negative: the drift files are damaged; re-run fetch-prj2048")
    return peaks


def shaking_intensities(acceleration_g: FloatArray, time_step: float) -> FloatArray:
    """ln PGA and ln Sa at SPECTRAL_PERIODS_S (5% damping), all in g."""
    pga = float(np.max(np.abs(acceleration_g)))
    spectral: list[float] = []
    for period in SPECTRAL_PERIODS_S:
        angular_frequency = 2 * math.pi / period
        peak = peak_oscillator_displacement(acceleration_g, time_step, angular_frequency, SPECTRUM_DAMPING_RATIO)
        spectral.append(angular_frequency**2 * peak)  # pseudo-spectral acceleration
    return np.log(np.array([pga, *spectral], dtype=np.float64))


def read_shaking(folder: Path) -> FloatArray:
    point_counts = np.loadtxt(folder / "GroundMotionInfo/GMNumPoints.txt", dtype=np.int64, ndmin=1)
    time_steps = np.loadtxt(folder / "GroundMotionInfo/GMtimeSteps.txt", dtype=np.float64, ndmin=1)
    shaking = np.empty((GROUND_MOTION_COUNT, len(SHAKING_INPUTS)), dtype=np.float64)
    for index in range(GROUND_MOTION_COUNT):
        path = folder / f"Histories/{index + 1}.txt"
        acceleration = np.loadtxt(path, dtype=np.float64)
        if acceleration.size != point_counts[index]:
            raise ValueError(f"{path}: {acceleration.size} samples, GMNumPoints.txt says {point_counts[index]}")
        shaking[index] = shaking_intensities(acceleration, float(time_steps[index]))
    return shaking


def load_prj2048(root: Path) -> Prj2048:
    building_ids, designs = read_designs(root / "DesignSummary.csv")
    return Prj2048(
        building_ids=building_ids,
        designs=designs,
        shaking=read_shaking(root / MOTION_FOLDER),
        peak_drifts=read_peak_drifts(root / DRIFT_FOLDER, building_ids),
    )
