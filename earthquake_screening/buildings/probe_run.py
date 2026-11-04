"""Slice 3's run: put every probe to slice 2's saved model, and judge the answers by the fixed pass rule (G4).

The rule (docs/building-surrogate/refusal-probes.md): the model is slice 2's, every
probe that should be refused is refused with a reason naming its field (or, for a
combination, saying it is unlike any training case), no refused probe gets a number,
and every probe that should be answered gets a finite drift and interval.
"""

import hashlib
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from earthquake_screening.buildings.dataset import FloatArray
from earthquake_screening.buildings.experiment import BuildingModel
from earthquake_screening.screening.envelope import TrainingEnvelope
from earthquake_screening.screening.probes import COMBINATION, Probe
from earthquake_screening.screening.refusal import assess
from earthquake_screening.screening.request import ScreenRequest

SLICE2_MODEL_SHA256 = "30e616b98e6e92f9bb0f779a6a039ad785f7a6ec32cd93108943010162c77def"
UNSEEN_COMBINATION_REASON = "unlike any training case"


@dataclass(frozen=True)
class ProbeOutcome:
    probe: Probe
    reasons: tuple[str, ...]
    drift: tuple[float, float, float] | None  # (prediction, low, high) as drift ratios; None when refused


@dataclass(frozen=True)
class ProbeRun:
    is_envelope_rebuilt: bool  # the fit rows rebuilt from the data give the saved envelope's ranges
    distance_limit: float
    combination_distances: dict[str, float]  # probe name -> distance to the nearest fit row
    outcomes: list[ProbeOutcome]


def load_slice2_model(path: Path) -> BuildingModel:
    """Unpickle the model only if it is slice 2's file, byte for byte."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SLICE2_MODEL_SHA256:
        raise ValueError(f"{path} is not slice 2's model (SHA-256 differs): run train-buildings on commit f906a35")
    return pickle.loads(content)  # noqa: S301 (this repo's own file, checked against a pinned hash)


def is_envelope_rebuilt(envelope: TrainingEnvelope, fit_inputs: FloatArray) -> bool:
    return bool(
        np.array_equal(fit_inputs.min(axis=0), envelope.minimum)
        and np.array_equal(fit_inputs.max(axis=0), envelope.maximum)
    )


def run_probe(probe: Probe, model: BuildingModel) -> ProbeOutcome:
    assessment = assess(probe.request, model.envelope)
    if assessment.inputs is None:
        return ProbeOutcome(probe, assessment.reasons, None)
    predicted, low, high = model.predict_interval(assessment.inputs[np.newaxis, :])
    drift = (math.exp(predicted[0]), math.exp(low[0]), math.exp(high[0]))
    return ProbeOutcome(probe, assessment.reasons, drift)


def _starts_with_field(reason: str, field: str) -> bool:
    return reason.split(" ", 1)[0].rstrip(":") == field


def is_as_expected(outcome: ProbeOutcome) -> bool:
    probe = outcome.probe
    if not probe.should_refuse:
        return outcome.drift is not None and not outcome.reasons and all(map(math.isfinite, outcome.drift))
    if outcome.drift is not None:
        return False
    if probe.field is None:
        return any(UNSEEN_COMBINATION_REASON in reason for reason in outcome.reasons)
    return any(_starts_with_field(reason, probe.field) for reason in outcome.reasons)


def run_probes(model: BuildingModel, fit_inputs: FloatArray, probes: list[Probe]) -> ProbeRun:
    envelope = model.envelope
    combination_distances = {
        probe.name: envelope.nearest_distance(ScreenRequest.model_validate(probe.request).as_inputs())
        for probe in probes
        if probe.kind == COMBINATION
    }
    return ProbeRun(
        is_envelope_rebuilt=is_envelope_rebuilt(envelope, fit_inputs),
        distance_limit=envelope.distance_limit,
        combination_distances=combination_distances,
        outcomes=[run_probe(probe, model) for probe in probes],
    )


def has_passed(run: ProbeRun) -> bool:
    """The model's hash is checked on loading; the rest of the rule is judged here."""
    return run.is_envelope_rebuilt and all(is_as_expected(outcome) for outcome in run.outcomes)
