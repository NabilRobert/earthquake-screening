"""The split: no frame in two parts, and held-out shaking never reaches the fit (requirement M1)."""

import dataclasses

import numpy as np

from earthquake_screening.buildings.dataset import STOREYS_COLUMN
from earthquake_screening.buildings.split import Split, is_split_clean, rows_for, split_prj2048
from tests.conftest import made_up_prj2048


def test_no_frame_is_in_two_parts() -> None:
    split = split_prj2048(made_up_prj2048())
    parts = (split.fit_frames, split.calibration_frames, split.test_frames)
    assert sum(part.size for part in parts) == 48
    assert np.unique(np.concatenate(parts)).size == 48
    assert is_split_clean(split)


def test_held_out_shaking_never_reaches_the_fit_rows() -> None:
    data = made_up_prj2048()
    split = split_prj2048(data)
    fit = rows_for(data, split.fit_frames, split.training_motions)
    assert np.intersect1d(fit.motions, split.held_out_motions).size == 0
    assert np.intersect1d(fit.frames, split.test_frames).size == 0


def test_a_rare_storey_count_appears_in_every_part() -> None:
    data = made_up_prj2048()
    designs = data.designs.copy()
    designs[:, STOREYS_COLUMN] = 1.0
    designs[:5, STOREYS_COLUMN] = 19.0  # 5 tall frames among 48: a random split often leaves a part without one
    rare = dataclasses.replace(data, designs=designs)
    for seed in range(20):
        split = split_prj2048(rare, seed=seed)
        for part in (split.fit_frames, split.calibration_frames, split.test_frames):
            assert 19.0 in designs[part, STOREYS_COLUMN]


def test_the_same_seed_gives_the_same_split() -> None:
    data = made_up_prj2048()
    first, second = split_prj2048(data, seed=5), split_prj2048(data, seed=5)
    assert np.array_equal(first.test_frames, second.test_frames)
    assert np.array_equal(first.held_out_motions, second.held_out_motions)


def test_a_frame_in_two_parts_makes_the_split_unclean() -> None:
    frames = np.arange(3, dtype=np.int64)
    leaky = Split(frames[:2], frames[1:2], frames[2:], frames[:2], frames[2:])
    assert not is_split_clean(leaky)


def test_a_motion_both_trained_and_held_out_makes_the_split_unclean() -> None:
    frames = np.arange(3, dtype=np.int64)
    leaky = Split(frames[:1], frames[1:2], frames[2:], frames[:2], frames[1:])
    assert not is_split_clean(leaky)


def test_rows_pair_each_frame_with_each_motion() -> None:
    data = made_up_prj2048()
    rows = rows_for(data, np.array([0, 4], dtype=np.int64), np.array([1, 2, 3], dtype=np.int64))
    assert rows.inputs.shape == (6, 12)
    assert rows.ln_drifts[-1] == np.log(data.peak_drifts[4, 3])
    assert np.array_equal(rows.inputs[-1, 6:], data.shaking[3])
