"""The probe suite (R3): screens that must be refused, and screens that must be answered.

Built from the model's fit rows only, as fixed in docs/building-surrogate/refusal-probes.md:
- a typical row (nearest the fit rows' median), answered as is;
- for each input and bound, the fit row at that bound nearest the typical row, moved
  1% inward (answered) or 1% outward (refused); counts stay at the bound or go one past;
- two shaking combinations no real record has, every value in range (refused);
- 21 absurd values put into the typical row (refused).
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from earthquake_screening.buildings.dataset import BUILDING_INPUTS, FloatArray
from earthquake_screening.screening.request import REQUEST_FIELDS, request_from_inputs

TYPICAL = "typical"
JUST_INSIDE = "just inside"
JUST_OUTSIDE = "just outside"
COMBINATION = "unseen combination"
ABSURD = "absurd"

COUNT_FIELDS = ("storeys", "bays")
NUDGE = 0.01  # "just": 1% of the bound, far beyond the envelope's 1e-9 rounding tolerance
SHAKING_FIELDS = REQUEST_FIELDS[len(BUILDING_INPUTS) :]  # pga_g and the five Sa values
# Absurd values, each put into the typical row in place of the field's value.
ABSURD_VALUES: tuple[tuple[str, object], ...] = (
    ("storeys", 0),
    ("storeys", -3),
    ("storeys", 2.5),
    ("storeys", "\u0661\u0664"),  # Arabic-Indic 14
    ("storeys", "\uff11\uff14"),  # full-width 14
    ("storeys", True),
    ("storeys", "1_4"),
    ("storeys", ""),
    ("storeys", "14 storeys"),
    ("storeys", 10**9),
    ("bay_width_ft", -30.0),
    ("bay_width_ft", float("nan")),
    ("bay_width_ft", "nan"),
    ("bay_width_ft", float("inf")),
    ("bay_width_ft", 1e308),
    ("story_height_ratio", 0.0),
    ("pga_g", 0.0),
    ("pga_g", -0.3),
    ("sa_1s_g", "inf"),
)


@dataclass(frozen=True)
class Probe:
    name: str
    kind: str
    request: Mapping[str, object]
    should_refuse: bool
    field: str | None  # the field a refusal reason must start with; None for a combination


def _standardised(fit_inputs: FloatArray) -> FloatArray:
    scale = np.where(fit_inputs.std(axis=0) > 0, fit_inputs.std(axis=0), 1.0)
    return (fit_inputs - fit_inputs.mean(axis=0)) / scale


def typical_row(fit_inputs: FloatArray) -> int:
    standardised = _standardised(fit_inputs)
    distances = np.linalg.norm(standardised - np.median(standardised, axis=0), axis=1)
    return int(np.argmin(distances))


def bound_row(fit_inputs: FloatArray, column: int, bound: float, typical: int) -> int:
    """The fit row with this input at this bound that is nearest the typical row."""
    standardised = _standardised(fit_inputs)
    distances: FloatArray = np.linalg.norm(standardised - standardised[typical : typical + 1], axis=1)
    is_at_bound = fit_inputs[:, column] == bound
    return int(np.argmin(np.where(is_at_bound, distances, np.inf)))


def _moved(value: float, field: str, is_minimum: bool, is_outward: bool) -> float:
    """The bound moved one step: one count, or 1% of the value."""
    goes_down = is_minimum == is_outward  # outward from the minimum, or inward from the maximum
    if field in COUNT_FIELDS:
        if not is_outward:
            return value  # a count just inside is the bound itself
        return value - 1 if goes_down else value + 1
    return value * (1 - NUDGE) if goes_down else value * (1 + NUDGE)


def boundary_probes(fit_inputs: FloatArray, typical: int) -> list[Probe]:
    probes: list[Probe] = []
    for column, field in enumerate(REQUEST_FIELDS):
        for side, bound in (("minimum", fit_inputs[:, column].min()), ("maximum", fit_inputs[:, column].max())):
            request = request_from_inputs(fit_inputs[bound_row(fit_inputs, column, float(bound), typical)])
            for kind, is_outward in ((JUST_INSIDE, False), (JUST_OUTSIDE, True)):
                moved = _moved(request[field], field, side == "minimum", is_outward)
                name = f"{field} {kind} its {side}: {moved:.4g}"
                probes.append(Probe(name, kind, {**request, field: moved}, is_outward, field))
    return probes


def combination_probes(fit_inputs: FloatArray, typical: int) -> list[Probe]:
    """Two spectra no real record has, on the typical row's design; each value is within its range."""
    request = request_from_inputs(fit_inputs[typical])
    lowest = request_from_inputs(fit_inputs.min(axis=0))
    highest = request_from_inputs(fit_inputs.max(axis=0))
    strong_pga_weak_spectrum = {field: lowest[field] for field in SHAKING_FIELDS} | {"pga_g": highest["pga_g"]}
    weak_short_strong_long = {field: lowest[field] for field in ("pga_g", "sa_0_2s_g", "sa_0_5s_g")} | {
        field: highest[field] for field in ("sa_2s_g", "sa_4s_g")
    }
    return [
        Probe(
            "PGA at its maximum, every Sa at its minimum", COMBINATION, request | strong_pga_weak_spectrum, True, None
        ),
        Probe(
            "PGA and short-period Sa at their minimum, long-period Sa at their maximum",
            COMBINATION,
            request | weak_short_strong_long,
            True,
            None,
        ),
    ]


def absurd_probes(fit_inputs: FloatArray, typical: int) -> list[Probe]:
    request: dict[str, object] = dict(request_from_inputs(fit_inputs[typical]))
    probes = [
        Probe(f"{field} = {value!r}", ABSURD, request | {field: value}, True, field) for field, value in ABSURD_VALUES
    ]
    without_storeys = {field: value for field, value in request.items() if field != "storeys"}
    probes.append(Probe("storeys left out", ABSURD, without_storeys, True, "storeys"))
    probes.append(Probe("an unknown field, storey = 5", ABSURD, request | {"storey": 5}, True, "storey"))
    return probes


def build_probes(fit_inputs: FloatArray) -> list[Probe]:
    """Every probe, in the order of the pass rule's table."""
    typical = typical_row(fit_inputs)
    typical_probe = Probe("the typical fit row", TYPICAL, dict(request_from_inputs(fit_inputs[typical])), False, None)
    return [
        typical_probe,
        *boundary_probes(fit_inputs, typical),
        *combination_probes(fit_inputs, typical),
        *absurd_probes(fit_inputs, typical),
    ]
