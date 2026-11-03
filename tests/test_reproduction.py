"""The pass rule, and a small end-to-end run on made-up records (a smoke test, not a result)."""

from earthquake_screening.reproduction.sainct2020 import PROJECTIONS, RowResult, has_passed, run_reproduction
from earthquake_screening.simulation.oscillator import Oscillator
from earthquake_screening.simulation.pool import build_pool
from earthquake_screening.simulation.synthesis import draw_synthetic_specs
from tests.conftest import made_up_fitted_records


def _rows(pga_100: float, pga_1000: float, others: float = 99.0) -> list[RowResult]:
    rows = [RowResult("pga", 100, (pga_100,) * 2), RowResult("pga", 1000, (pga_1000,) * 2)]
    rows += [
        RowResult(projection, budget, (others,) * 2)
        for projection in ("score", "linear_peak")
        for budget in (100, 1000)
    ]
    return rows


def test_inside_the_papers_pga_band_at_both_budgets_passes_whatever_the_other_rows() -> None:
    # Paper: 2.6 +/- 1.0 at 100 and 1.6 +/- 0.3 at 1,000.
    assert has_passed(_rows(pga_100=3.5, pga_1000=1.4))


def test_outside_the_band_at_either_budget_is_a_miss() -> None:
    assert not has_passed(_rows(pga_100=3.7, pga_1000=1.6))
    assert not has_passed(_rows(pga_100=2.6, pga_1000=2.0))


def test_a_small_run_gives_a_finite_error_for_every_curve_and_budget() -> None:
    specs = draw_synthetic_specs(made_up_fitted_records(30, seed=0), 600, seed=1)
    pool = build_pool(specs, Oscillator())
    rows = run_reproduction(pool, seed=2, test_case_count=2, budgets=(10, 40))
    assert {(row.projection, row.budget) for row in rows} == {(p, b) for p in PROJECTIONS for b in (10, 40)}
    assert all(0 <= error < 100 for row in rows for error in row.errors_percent)
