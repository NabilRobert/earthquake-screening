"""Reproduce Table 2 of Sainct et al. (2020) and judge it against a pass rule fixed before running.

The paper's 4-D linear SVM, started from 20 random pairs, gives fragility errors
(Delta L2, mean +/- standard deviation over the 20) for curves against the SVM score,
PGA and L, after 100 and after 1,000 labelled records. We run the same 20-start
experiment on our pool and compare.

Pass rule (docs/reproduction/sainct2020.md): our mean error against PGA falls inside the
paper's mean +/- one standard deviation at both 100 and 1,000 records. The other rows
are reported alongside, not judged.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

import numpy as np

from earthquake_screening.simulation.ground_motion import PARAMETER_NAMES
from earthquake_screening.simulation.pool import INTENSITY_NAMES, Pool
from earthquake_screening.surrogate.active_learning import (
    BoolArray,
    FloatArray,
    IndexArray,
    run_active_learning,
    standardised_features,
    starting_pair,
)
from earthquake_screening.surrogate.fragility import failure_probabilities, fragility_error, group_by_value

BUDGETS = (100, 1000)
TEST_CASE_COUNT = 20
# The paper doesn't say how many k-means groups it used. More groups give a larger
# error (each group's reference rate is noisier), so we chose a high count, 50, before
# running, to avoid flattering ourselves.
FRAGILITY_GROUP_COUNT = 50
PROJECTIONS = ("score", "pga", "linear_peak")

# Table 2 of the paper, 4-D linear SVM: (mean, standard deviation) of Delta L2, in percent.
PAPER_TABLE_2: dict[tuple[str, int], tuple[float, float]] = {
    ("score", 100): (3.8, 1.6),
    ("score", 1000): (1.7, 0.6),
    ("pga", 100): (2.6, 1.0),
    ("pga", 1000): (1.6, 0.3),
    ("linear_peak", 100): (2.8, 0.9),
    ("linear_peak", 1000): (1.4, 0.4),
}
JUDGED_PROJECTION = "pga"


@dataclass(frozen=True)
class RowResult:
    projection: str
    budget: int
    errors_percent: tuple[float, ...]  # one per test case

    @property
    def mean(self) -> float:
        return float(np.mean(self.errors_percent))

    @property
    def standard_deviation(self) -> float:
        return float(np.std(self.errors_percent, ddof=1))

    @property
    def is_inside_paper_band(self) -> bool:
        paper_mean, paper_deviation = PAPER_TABLE_2[(self.projection, self.budget)]
        return abs(self.mean - paper_mean) <= paper_deviation


def has_passed(rows: list[RowResult]) -> bool:
    return all(row.is_inside_paper_band for row in rows if row.projection == JUDGED_PROJECTION)


def four_feature_matrix(pool: Pool, rows: IndexArray) -> FloatArray:
    """(L, PGA, PGV, w0), the paper's 4-D input space (section 4.1)."""
    columns = [
        pool.intensity[rows, INTENSITY_NAMES.index("linear_peak")],
        pool.intensity[rows, INTENSITY_NAMES.index("pga")],
        pool.intensity[rows, INTENSITY_NAMES.index("pgv")],
        pool.parameters[rows, PARAMETER_NAMES.index("start_frequency")],
    ]
    return np.column_stack(columns)


@dataclass(frozen=True)
class _Experiment:
    """What every test case shares: the pool's features, labels and fixed groups."""

    features: FloatArray
    has_failed: BoolArray
    groups_by_measure: dict[str, IndexArray]
    budgets: tuple[int, ...]
    seed: int


def _run_test_case(experiment: _Experiment, start: IndexArray) -> dict[tuple[str, int], float]:
    """One start pair: Delta L2 in percent for each projection and budget."""
    errors: dict[tuple[str, int], float] = {}
    snapshots = run_active_learning(experiment.features, experiment.has_failed, start, experiment.budgets)
    for budget, snapshot in snapshots.items():
        probabilities = failure_probabilities(snapshot.scores, snapshot.labelled_indices, experiment.has_failed)
        score_groups = group_by_value(snapshot.scores, FRAGILITY_GROUP_COUNT, experiment.seed)
        groups = {**experiment.groups_by_measure, "score": score_groups}
        for projection in PROJECTIONS:
            error = fragility_error(probabilities, experiment.has_failed, groups[projection])
            errors[(projection, budget)] = 100 * error
    return errors


def run_reproduction(
    pool: Pool, seed: int, test_case_count: int = TEST_CASE_COUNT, budgets: tuple[int, ...] = BUDGETS
) -> list[RowResult]:
    """The 20-start experiment, test cases in parallel; one row per projection and budget."""
    rows = np.flatnonzero(pool.labelled_rows)
    raw_features = four_feature_matrix(pool, rows)
    pga, linear_peak = raw_features[:, 1], raw_features[:, 0]
    experiment = _Experiment(
        features=standardised_features(raw_features),
        has_failed=pool.has_failed[rows],
        groups_by_measure={
            "pga": group_by_value(pga, FRAGILITY_GROUP_COUNT, seed),
            "linear_peak": group_by_value(linear_peak, FRAGILITY_GROUP_COUNT, seed),
        },
        budgets=budgets,
        seed=seed,
    )
    # Start pairs are drawn here, in order, so the result doesn't depend on scheduling.
    rng = np.random.default_rng(seed)
    starts = [starting_pair(pga, linear_peak, experiment.has_failed, rng) for _ in range(test_case_count)]
    with ProcessPoolExecutor() as executor:
        per_case = list(executor.map(partial(_run_test_case, experiment), starts))
    return [
        RowResult(projection, budget, tuple(case[(projection, budget)] for case in per_case))
        for projection in PROJECTIONS
        for budget in budgets
    ]
