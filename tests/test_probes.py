"""The probe suite is built as fixed in docs/building-surrogate/refusal-probes.md, from fit rows only."""

from collections import Counter

import numpy as np
import pytest

from earthquake_screening.buildings.split import rows_for, split_prj2048
from earthquake_screening.screening.probes import (
    ABSURD,
    COMBINATION,
    JUST_INSIDE,
    JUST_OUTSIDE,
    TYPICAL,
    Probe,
    bound_row,
    build_probes,
    typical_row,
)
from earthquake_screening.screening.request import REQUEST_FIELDS, request_from_inputs
from tests.conftest import made_up_prj2048


@pytest.fixture(scope="module")
def fit_inputs() -> np.ndarray:
    data = made_up_prj2048()
    split = split_prj2048(data)
    return rows_for(data, split.fit_frames, split.training_motions).inputs


def boundary(probes: list[Probe], kind: str, field: str, side: str) -> Probe:
    return next(probe for probe in probes if probe.kind == kind and probe.name.startswith(f"{field} {kind} its {side}"))


def test_the_suite_has_the_fixed_number_of_each_kind(fit_inputs: np.ndarray) -> None:
    kinds = Counter(probe.kind for probe in build_probes(fit_inputs))
    assert kinds == {TYPICAL: 1, JUST_INSIDE: 24, JUST_OUTSIDE: 24, COMBINATION: 2, ABSURD: 21}


def test_only_the_typical_and_just_inside_probes_expect_an_answer(fit_inputs: np.ndarray) -> None:
    for probe in build_probes(fit_inputs):
        assert probe.should_refuse is (probe.kind not in (TYPICAL, JUST_INSIDE))


@pytest.mark.parametrize(
    ("kind", "side", "expected_storeys"), [(JUST_INSIDE, "minimum", 1), (JUST_OUTSIDE, "minimum", 0),
                                           (JUST_INSIDE, "maximum", 9), (JUST_OUTSIDE, "maximum", 10)]
)  # fmt: skip
def test_a_count_stays_at_its_bound_inside_and_goes_one_past_outside(
    fit_inputs: np.ndarray, kind: str, side: str, expected_storeys: int
) -> None:
    assert boundary(build_probes(fit_inputs), kind, "storeys", side).request["storeys"] == expected_storeys


@pytest.mark.parametrize(
    ("kind", "side", "factor"), [(JUST_INSIDE, "minimum", 1.01), (JUST_OUTSIDE, "minimum", 0.99),
                                 (JUST_INSIDE, "maximum", 0.99), (JUST_OUTSIDE, "maximum", 1.01)]
)  # fmt: skip
def test_other_values_move_one_percent_from_their_bound(
    fit_inputs: np.ndarray, kind: str, side: str, factor: float
) -> None:
    bound = np.exp(fit_inputs[:, REQUEST_FIELDS.index("sa_1s_g")])
    expected = factor * (bound.min() if side == "minimum" else bound.max())
    assert boundary(build_probes(fit_inputs), kind, "sa_1s_g", side).request["sa_1s_g"] == pytest.approx(expected)


def test_a_boundary_probe_changes_only_its_own_field_from_its_bound_row(fit_inputs: np.ndarray) -> None:
    column = REQUEST_FIELDS.index("bay_width_ft")
    typical = typical_row(fit_inputs)
    row = request_from_inputs(fit_inputs[bound_row(fit_inputs, column, float(fit_inputs[:, column].max()), typical)])
    probe = boundary(build_probes(fit_inputs), JUST_OUTSIDE, "bay_width_ft", "maximum")
    assert [field for field in REQUEST_FIELDS if probe.request[field] != row[field]] == ["bay_width_ft"]


def test_the_bound_row_is_the_one_at_the_bound_nearest_the_typical_row() -> None:
    fit_inputs = np.zeros((4, len(REQUEST_FIELDS)))
    fit_inputs[:, 0] = [1.0, 2.0, 2.0, 2.0]
    fit_inputs[:, 1] = [0.0, 5.0, 1.0, -3.0]
    assert bound_row(fit_inputs, column=0, bound=2.0, typical=0) == 2


def test_the_typical_row_is_the_one_nearest_the_median() -> None:
    fit_inputs = np.zeros((5, len(REQUEST_FIELDS)))
    fit_inputs[:, 0] = [9.0, 0.0, 4.0, 1.0, 5.0]  # median 4
    assert typical_row(fit_inputs) == 2


def test_the_first_combination_has_the_strongest_pga_and_the_weakest_sa_everywhere(fit_inputs: np.ndarray) -> None:
    probe = next(probe for probe in build_probes(fit_inputs) if probe.kind == COMBINATION)
    lowest, highest = request_from_inputs(fit_inputs.min(axis=0)), request_from_inputs(fit_inputs.max(axis=0))
    assert probe.request["pga_g"] == highest["pga_g"]
    assert [probe.request[field] for field in REQUEST_FIELDS[7:]] == [lowest[field] for field in REQUEST_FIELDS[7:]]


def test_the_combination_probes_keep_every_value_within_its_range(fit_inputs: np.ndarray) -> None:
    low, high = np.exp(fit_inputs.min(axis=0)), np.exp(fit_inputs.max(axis=0))
    for probe in build_probes(fit_inputs):
        if probe.kind != COMBINATION:
            continue
        for column, field in enumerate(REQUEST_FIELDS[6:], start=6):
            value = probe.request[field]
            assert isinstance(value, float)
            assert low[column] * (1 - 1e-9) <= value <= high[column] * (1 + 1e-9)
