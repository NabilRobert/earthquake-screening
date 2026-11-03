"""The Hazus-style class baseline: Hazus's S1 height classes, one line per class."""

import numpy as np
import pytest

from earthquake_screening.buildings.dataset import INPUT_NAMES, STOREYS_COLUMN
from earthquake_screening.surrogate.class_baseline import ClassBaseline, height_class


@pytest.mark.parametrize(("storeys", "expected"), [(1, "low"), (3, "low"), (4, "mid"), (7, "mid"), (8, "high")])
def test_height_classes_follow_hazus_s1_boundaries(storeys: int, expected: str) -> None:
    assert height_class(storeys) == expected


def test_zero_storeys_fits_no_class() -> None:
    with pytest.raises(ValueError, match="no Hazus height class"):
        height_class(0)


def test_baseline_recovers_each_class_line_and_its_period() -> None:
    rng = np.random.default_rng(0)
    inputs = rng.normal(0.0, 1.0, (600, len(INPUT_NAMES)))
    inputs[:, STOREYS_COLUMN] = np.repeat([1.0, 5.0, 14.0], 200)
    sa_1s = INPUT_NAMES.index("ln_sa_1s_g")
    sa_2s = INPUT_NAMES.index("ln_sa_2s_g")
    ln_drifts = np.where(inputs[:, STOREYS_COLUMN] < 8, -5.0 + 1.0 * inputs[:, sa_1s], -4.0 + 0.8 * inputs[:, sa_2s])
    baseline = ClassBaseline.fit(inputs, ln_drifts)
    assert baseline.lines["low"].spectral_column == sa_1s
    assert baseline.lines["high"].spectral_column == sa_2s
    assert baseline.lines["high"].slope == pytest.approx(0.8)
    assert np.allclose(baseline.predict(inputs), ln_drifts)
