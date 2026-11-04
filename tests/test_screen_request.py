"""A screen's values are refused, naming the field, unless each is plainly a possible number."""

import math

import numpy as np
import pytest

from earthquake_screening.buildings.dataset import INPUT_NAMES
from earthquake_screening.screening.envelope import TrainingEnvelope
from earthquake_screening.screening.refusal import assess
from earthquake_screening.screening.request import REQUEST_FIELDS, ScreenRequest, request_from_inputs

PLAIN: dict[str, object] = {
    "storeys": 5,
    "bay_width_ft": 30.0,
    "story_height_ratio": 1.5,
    "bays": 3,
    "floor_dead_load_psf": 80.0,
    "roof_dead_load_psf": 67.5,
    "pga_g": 0.2,
    "sa_0_2s_g": 0.5,
    "sa_0_5s_g": 0.4,
    "sa_1s_g": 0.2,
    "sa_2s_g": 0.08,
    "sa_4s_g": 0.03,
}


def test_a_plain_request_becomes_the_model_inputs_with_shaking_as_ln_g() -> None:
    inputs = ScreenRequest.model_validate(PLAIN).as_inputs()
    assert inputs[:6].tolist() == [5.0, 30.0, 1.5, 3.0, 80.0, 67.5]
    assert inputs[INPUT_NAMES.index("ln_sa_1s_g")] == pytest.approx(math.log(0.2))


def test_turning_inputs_back_into_a_request_round_trips() -> None:
    inputs = ScreenRequest.model_validate(PLAIN).as_inputs()
    assert request_from_inputs(inputs) == pytest.approx(PLAIN)


@pytest.mark.parametrize("text", ["14", " 14 ", "14.0", "+14"])
def test_numbers_written_with_digits_0_to_9_are_read(text: str) -> None:
    assert ScreenRequest.model_validate({**PLAIN, "storeys": text}).storeys == 14


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("storeys", "\u0661\u0664"),  # Arabic-Indic 14, which Python's int() reads
        ("storeys", "\uff11\uff14"),  # full-width 14
        ("storeys", "1_4"),  # Pydantic reads this as 14
        ("storeys", True),  # Pydantic reads this as 1
        ("storeys", ""),
        ("storeys", "14 storeys"),
        ("storeys", 0),
        ("storeys", 2.5),
        ("bays", -1),
        ("bay_width_ft", float("nan")),
        ("bay_width_ft", "nan"),
        ("bay_width_ft", float("inf")),
        ("bay_width_ft", -30.0),
        ("story_height_ratio", 0.0),
        ("pga_g", 0.0),
        ("sa_4s_g", "inf"),
    ],
)
def test_a_value_that_is_not_plainly_possible_is_refused_naming_its_field(field: str, value: object) -> None:
    envelope = everything_envelope()
    assessment = assess({**PLAIN, field: value}, envelope)
    assert assessment.is_refused
    assert len(assessment.reasons) == 1
    assert assessment.reasons[0].startswith(f"{field}: ")


def test_a_missing_field_and_an_unknown_field_are_both_named() -> None:
    raw = {field: value for field, value in PLAIN.items() if field != "bays"} | {"storey": 5}
    reasons = assess(raw, everything_envelope()).reasons
    assert sorted(reason.split(":")[0] for reason in reasons) == ["bays", "storey"]


def everything_envelope() -> TrainingEnvelope:
    """An envelope around PLAIN's inputs, with room on every side."""
    centre = ScreenRequest.model_validate(PLAIN).as_inputs()
    rng = np.random.default_rng(0)
    rows = centre + rng.normal(0.0, 1.0, (200, centre.size)) * np.maximum(1.0, np.abs(centre))
    return TrainingEnvelope.from_training(rows, centre[np.newaxis, :])


def test_an_accepted_screen_carries_its_inputs_and_no_reasons() -> None:
    assessment = assess(PLAIN, everything_envelope())
    assert not assessment.is_refused
    assert assessment.reasons == ()
    assert assessment.inputs is not None and assessment.inputs.size == len(REQUEST_FIELDS)


def test_a_screen_the_envelope_refuses_carries_no_inputs() -> None:
    assessment = assess({**PLAIN, "storeys": 10**9}, everything_envelope())
    assert assessment.inputs is None
    assert assessment.reasons[0].startswith("storeys = 1e+09 is outside the training range")


def test_the_training_maximum_typed_in_g_is_accepted_despite_rounding() -> None:
    # exp then log moves this value up by one rounding step (found by search; about 7% of values do this).
    ln_maximum = -0.3284793480238726
    assert math.log(math.exp(ln_maximum)) > ln_maximum
    rows = np.tile(ScreenRequest.model_validate(PLAIN).as_inputs(), (20, 1))
    rows[:, INPUT_NAMES.index("ln_pga_g")] = np.linspace(ln_maximum - 1.0, ln_maximum, 20)
    envelope = TrainingEnvelope.from_training(rows, rows)
    assert assess(request_from_inputs(rows[-1]), envelope).reasons == ()


def test_a_request_that_is_not_a_set_of_named_values_is_refused_as_a_whole() -> None:
    reasons = assess([5, 30.0], everything_envelope()).reasons  # type: ignore[arg-type]  # as an agent might send
    assert reasons == ("request: Input should be a valid dictionary or instance of ScreenRequest (got [5, 30.0])",)
