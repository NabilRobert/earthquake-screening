"""What a user asks to screen: twelve named values in the units they know, checked before the envelope sees them.

A screen must never guess at a value (R2), so anything that isn't plainly a possible
number is refused here with the field's name: digits other than 0-9 (Python reads
"١٤" as 14), Pydantic's "1_4" and true/false, NaN, infinity, counts that aren't whole,
and values at or below 0. The envelope then checks the training range, on the
model's inputs: shaking becomes ln g (docs/building-surrogate/refusal-probes.md).
"""

import math
import re
from typing import Annotated

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from earthquake_screening.buildings.dataset import INPUT_NAMES, SHAKING_INPUTS, FloatArray

# An optional sign, digits 0-9 with an optional decimal point, an optional exponent.
# [0-9], not \d: \d also matches other scripts' digits.
PLAIN_NUMBER = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?")


def _plain_number(value: object) -> object:
    if isinstance(value, bool):  # Pydantic would read true as 1
        raise ValueError("expected a number, not true or false")
    if isinstance(value, str) and not PLAIN_NUMBER.fullmatch(value.strip()):
        raise ValueError("write it as a plain number with the digits 0-9, for example 14 or 0.35")
    return value


Count = Annotated[int, BeforeValidator(_plain_number), Field(ge=1)]
Positive = Annotated[float, BeforeValidator(_plain_number), Field(gt=0)]


class ScreenRequest(BaseModel):
    """One building under one shaking. Field order matches the model's INPUT_NAMES."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    storeys: Count
    bay_width_ft: Positive
    story_height_ratio: Positive
    bays: Count
    floor_dead_load_psf: Positive
    roof_dead_load_psf: Positive
    pga_g: Positive
    sa_0_2s_g: Positive
    sa_0_5s_g: Positive
    sa_1s_g: Positive
    sa_2s_g: Positive
    sa_4s_g: Positive

    def as_inputs(self) -> FloatArray:
        """The model's input row: the design as given, the shaking as ln g."""
        values = [float(getattr(self, field)) for field in REQUEST_FIELDS]
        return np.array(
            [math.log(value) if is_logged else value for value, is_logged in zip(values, IS_LOGGED, strict=True)],
            dtype=np.float64,
        )


REQUEST_FIELDS: tuple[str, ...] = tuple(ScreenRequest.model_fields)
IS_LOGGED = tuple(name in SHAKING_INPUTS for name in INPUT_NAMES)
assert len(REQUEST_FIELDS) == len(INPUT_NAMES)  # one field per model input, in the same order


def user_value(column: int, model_value: float) -> float:
    """A model input back in the user's units: g for shaking, as given for the design."""
    return math.exp(model_value) if IS_LOGGED[column] else model_value


def request_from_inputs(inputs: FloatArray) -> dict[str, float]:
    """The request a user would type for a model input row."""
    return {field: user_value(column, float(inputs[column])) for column, field in enumerate(REQUEST_FIELDS)}
