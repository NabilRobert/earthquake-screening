"""Reading PRJ-2048: designs, peak drifts and the shaking, from small files written here."""

import math
from pathlib import Path

import numpy as np
import pytest

from earthquake_screening.buildings import dataset
from earthquake_screening.buildings.dataset import (
    GROUND_MOTION_COUNT,
    SHAKING_INPUTS,
    read_designs,
    read_peak_drifts,
    read_shaking,
    shaking_intensities,
)

DESIGN_HEADER = (
    "building ID,number of story,bay width,story height ratio,number of bay,floor dead load,roof dead load,"
    "column beam ratio\n"
)


def write_drifts(folder: Path, building_id: int, storey_drifts: list[list[float]]) -> None:
    header = ",".join(f"GM{index + 1}" for index in range(len(storey_drifts[0])))
    rows = "\n".join(",".join(str(value) for value in storey) for storey in storey_drifts)
    (folder / f"Building_{building_id}_PSDR.csv").write_text(f"{header}\n{rows}\n")


def test_designs_keep_prj2048_ids_and_drop_the_constant_column(tmp_path: Path) -> None:
    path = tmp_path / "DesignSummary.csv"
    path.write_text(DESIGN_HEADER + "7.0,5.0,30.0,1.5,3.0,80.0,67.5,1.0\n")
    building_ids, designs = read_designs(path)
    assert building_ids.tolist() == [7]
    assert designs.tolist() == [[5.0, 30.0, 1.5, 3.0, 80.0, 67.5]]


def test_peak_drift_is_the_largest_over_storeys(tmp_path: Path) -> None:
    lower = [0.01] * GROUND_MOTION_COUNT
    upper = [0.02] * (GROUND_MOTION_COUNT - 1) + [0.005]
    write_drifts(tmp_path, 3, [lower, upper])
    peaks = read_peak_drifts(tmp_path, np.array([3], dtype=np.int64))
    assert peaks[0, 0] == 0.02
    assert peaks[0, -1] == 0.01


def test_drift_file_with_a_missing_ground_motion_is_rejected(tmp_path: Path) -> None:
    write_drifts(tmp_path, 3, [[0.01] * (GROUND_MOTION_COUNT - 1)])
    with pytest.raises(ValueError, match="239 ground motions"):
        read_peak_drifts(tmp_path, np.array([3], dtype=np.int64))


def test_zero_drift_is_rejected(tmp_path: Path) -> None:
    write_drifts(tmp_path, 3, [[0.0] + [0.01] * (GROUND_MOTION_COUNT - 1)])
    with pytest.raises(ValueError, match="zero or negative"):
        read_peak_drifts(tmp_path, np.array([3], dtype=np.int64))


def test_slow_shaking_has_spectral_acceleration_equal_to_its_peak() -> None:
    # A 20 s sine is quasi-static for every oscillator up to 4 s: Sa ~ PGA, within 5%.
    time = np.arange(0.0, 40.0, 0.01)
    acceleration = 0.3 * np.sin(2 * math.pi * time / 20.0)
    intensities = np.exp(shaking_intensities(acceleration, 0.01))
    assert intensities[0] == pytest.approx(0.3, rel=1e-3)
    assert intensities[SHAKING_INPUTS.index("ln_sa_0.2s_g")] == pytest.approx(0.3, rel=0.01)
    assert intensities[SHAKING_INPUTS.index("ln_sa_4s_g")] == pytest.approx(0.3, rel=0.05)


def test_history_shorter_than_its_stated_length_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset, "GROUND_MOTION_COUNT", 1)
    (tmp_path / "GroundMotionInfo").mkdir()
    (tmp_path / "Histories").mkdir()
    (tmp_path / "GroundMotionInfo/GMNumPoints.txt").write_text("4\n")
    (tmp_path / "GroundMotionInfo/GMtimeSteps.txt").write_text(".0200\n")
    (tmp_path / "Histories/1.txt").write_text("0.1\n-0.2\n0.05\n")
    with pytest.raises(ValueError, match="3 samples"):
        read_shaking(tmp_path)
