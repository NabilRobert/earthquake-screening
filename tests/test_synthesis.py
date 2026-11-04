"""Drawing synthetic parameter sets around the fitted records."""

from pathlib import Path

import pytest

from earthquake_screening.simulation.synthesis import draw_synthetic_specs, load_fitted, save_fitted
from tests.conftest import made_up_fitted_records


def test_draws_are_repeatable_from_the_seed() -> None:
    fitted = made_up_fitted_records(20, seed=0)
    assert draw_synthetic_specs(fitted, 50, seed=3) == draw_synthetic_specs(fitted, 50, seed=3)


def test_each_draw_keeps_its_parents_duration_and_has_its_own_seed() -> None:
    fitted = made_up_fitted_records(20, seed=0)
    durations = {record.record_id: record.duration_s for record in fitted}
    specs = draw_synthetic_specs(fitted, 200, seed=3)
    assert all(spec.duration_s == durations[spec.parent_record_id] for spec in specs)
    assert len({spec.seed for spec in specs}) == len(specs)


def test_fewer_than_two_fitted_records_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        draw_synthetic_specs(made_up_fitted_records(1, seed=0), 10, seed=3)


def test_fitted_records_round_trip_through_json(tmp_path: Path) -> None:
    fitted = made_up_fitted_records(5, seed=0)
    path = tmp_path / "fitted.json"
    save_fitted(fitted, path)
    assert load_fitted(path) == fitted
