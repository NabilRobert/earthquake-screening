"""Split PRJ-2048 by frame and by ground motion, so no test frame or test shaking is ever trained on.

Frames go to fit, calibration or test, stratified by storey count. A share of the
ground motions is held out from the fit entirely. Calibration and test rows both pair
unseen frames with held-out shaking, so their errors are exchangeable, which the
conformal intervals need (docs/building-surrogate/prj2048.md).
"""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from earthquake_screening.buildings.dataset import STOREYS_COLUMN, FloatArray, IntArray, Prj2048

SPLIT_SEED = 2048
TEST_SHARE = 0.20
CALIBRATION_SHARE = 0.15
HELD_OUT_MOTION_SHARE = 0.20


@dataclass(frozen=True)
class Split:
    """Row indices into Prj2048.designs (frames) and Prj2048.shaking (ground motions)."""

    fit_frames: IntArray
    calibration_frames: IntArray
    test_frames: IntArray
    training_motions: IntArray
    held_out_motions: IntArray


@dataclass(frozen=True)
class Rows:
    inputs: FloatArray  # one row per (frame, ground motion): INPUT_NAMES
    ln_drifts: FloatArray
    frames: IntArray  # which frame each row came from, for grouping
    motions: IntArray


def split_prj2048(data: Prj2048, seed: int = SPLIT_SEED) -> Split:
    frames = np.arange(data.designs.shape[0], dtype=np.int64)
    storeys = data.designs[:, STOREYS_COLUMN]
    rest, test = train_test_split(frames, test_size=TEST_SHARE, stratify=storeys, random_state=seed)
    fit, calibration = train_test_split(
        rest, test_size=CALIBRATION_SHARE / (1 - TEST_SHARE), stratify=storeys[rest], random_state=seed
    )
    motions = np.arange(data.shaking.shape[0], dtype=np.int64)
    training, held_out = train_test_split(motions, test_size=HELD_OUT_MOTION_SHARE, random_state=seed)
    return Split(
        fit_frames=np.sort(fit),
        calibration_frames=np.sort(calibration),
        test_frames=np.sort(test),
        training_motions=np.sort(training),
        held_out_motions=np.sort(held_out),
    )


def rows_for(data: Prj2048, frames: IntArray, motions: IntArray) -> Rows:
    """Every pairing of the given frames with the given ground motions."""
    frame_grid, motion_grid = (grid.ravel() for grid in np.meshgrid(frames, motions, indexing="ij"))
    inputs = np.hstack([data.designs[frame_grid], data.shaking[motion_grid]])
    return Rows(
        inputs=inputs,
        ln_drifts=np.log(data.peak_drifts[frame_grid, motion_grid]),
        frames=frame_grid,
        motions=motion_grid,
    )


def is_split_clean(split: Split) -> bool:
    """No frame in two parts, and no held-out ground motion among the training ones."""
    frame_parts = (split.fit_frames, split.calibration_frames, split.test_frames)
    frames_used = np.concatenate(frame_parts)
    motions_used = np.concatenate((split.training_motions, split.held_out_motions))
    return bool(np.unique(frames_used).size == frames_used.size and np.unique(motions_used).size == motions_used.size)
