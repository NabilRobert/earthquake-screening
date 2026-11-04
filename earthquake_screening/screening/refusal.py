"""Decide whether a screen may be answered: the request's checks, then the training envelope (ADR-009).

A refused screen carries its reasons and no model inputs, so nothing downstream can
turn it into a number.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from earthquake_screening.buildings.dataset import FloatArray
from earthquake_screening.screening.envelope import TrainingEnvelope
from earthquake_screening.screening.request import ScreenRequest


@dataclass(frozen=True)
class Assessment:
    inputs: FloatArray | None  # the model's input row, or None when refused
    reasons: tuple[str, ...]  # empty when accepted

    @property
    def is_refused(self) -> bool:
        return self.inputs is None


def describe_error(error: ErrorDetails) -> str:
    """One Pydantic error as a reason that starts with the field's name ("request" when it is the whole thing)."""
    field = ".".join(str(part) for part in error["loc"]) or "request"
    if error["type"] == "missing":
        return f"{field}: missing; give a value"
    return f"{field}: {error['msg']} (got {error['input']!r})"


def assess(raw: Mapping[str, object], envelope: TrainingEnvelope) -> Assessment:
    try:
        request = ScreenRequest.model_validate(raw)
    except ValidationError as error:
        return Assessment(None, tuple(describe_error(detail) for detail in error.errors()))
    inputs = request.as_inputs()
    reasons = envelope.refusal_reasons(inputs)
    return Assessment(None if reasons else inputs, tuple(reasons))
