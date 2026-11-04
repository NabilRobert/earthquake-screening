"""Slice 3's pass rule: refused probes name their field and get no number; answered probes get a finite one."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from mapie.regression import SplitConformalRegressor

from earthquake_screening.buildings.experiment import CONFIDENCE_LEVEL, BuildingModel
from earthquake_screening.buildings.probe_report import format_probe_report
from earthquake_screening.buildings.probe_run import (
    ProbeOutcome,
    ProbeRun,
    has_passed,
    is_as_expected,
    is_envelope_rebuilt,
    load_slice2_model,
    run_probes,
)
from earthquake_screening.buildings.split import rows_for, split_prj2048
from earthquake_screening.screening.envelope import TrainingEnvelope
from earthquake_screening.screening.probes import ABSURD, COMBINATION, JUST_INSIDE, TYPICAL, Probe, build_probes
from earthquake_screening.surrogate.building_model import BOOSTING, fit_candidate
from tests.conftest import made_up_prj2048

ANSWER = (0.01, 0.006, 0.016)


def outcome(should_refuse: bool, reasons: tuple[str, ...], drift: tuple[float, float, float] | None,
            field: str | None = "storeys") -> ProbeOutcome:  # fmt: skip
    kind = COMBINATION if field is None else ABSURD if should_refuse else JUST_INSIDE
    return ProbeOutcome(Probe("a probe", kind, {}, should_refuse, field), reasons, drift)


def test_an_answer_where_one_is_expected_is_as_expected() -> None:
    assert is_as_expected(outcome(False, (), ANSWER))


def test_a_non_finite_answer_is_not_an_answer() -> None:
    assert not is_as_expected(outcome(False, (), (math.nan, 0.006, 0.016)))
    assert not is_as_expected(outcome(False, (), (0.01, 0.006, math.inf)))


def test_a_refusal_where_an_answer_is_expected_is_a_miss() -> None:
    assert not is_as_expected(outcome(False, ("storeys = 20 is outside the training range 1 to 19",), None))


def test_a_refusal_must_name_the_probes_field() -> None:
    assert is_as_expected(outcome(True, ("storeys = 20 is outside the training range 1 to 19",), None))
    assert is_as_expected(outcome(True, ("storeys: missing; give a value",), None))
    assert not is_as_expected(outcome(True, ("bays = 9 is outside the training range 1 to 5",), None))
    assert not is_as_expected(outcome(True, ("storeys_total: Extra inputs are not permitted",), None))


def test_a_refused_probe_that_also_gets_a_number_is_a_miss() -> None:
    assert not is_as_expected(outcome(True, ("storeys: missing; give a value",), ANSWER))


def test_a_combination_must_be_refused_as_unlike_any_training_case() -> None:
    reason = "this combination of inputs is unlike any training case (distance 3.10, limit 1.20)"
    assert is_as_expected(outcome(True, (reason,), None, field=None))
    assert not is_as_expected(outcome(True, ("pga_g = 2 is outside the training range 0.03 to 0.78",), None, None))


def test_a_model_file_other_than_slice_2s_is_never_unpickled(tmp_path: Path) -> None:
    other = tmp_path / "model.pkl"
    other.write_bytes(b"not slice 2's model")
    with pytest.raises(ValueError, match="is not slice 2's model"):
        load_slice2_model(other)


def made_up_model() -> tuple[BuildingModel, np.ndarray]:
    """A boosting model on made-up data, trained as slice 2 trains (without its slow candidate comparison)."""
    data = made_up_prj2048(motion_count=40)
    split = split_prj2048(data)
    fit = rows_for(data, split.fit_frames, split.training_motions)
    calibration = rows_for(data, split.calibration_frames, split.held_out_motions)
    regressor = fit_candidate(BOOSTING, fit.inputs, fit.ln_drifts, seed=0)
    conformal = SplitConformalRegressor(regressor, confidence_level=CONFIDENCE_LEVEL, prefit=True)
    conformal.conformalize(calibration.inputs, calibration.ln_drifts)
    envelope = TrainingEnvelope.from_training(fit.inputs, calibration.inputs)
    return BuildingModel(BOOSTING, conformal, envelope), fit.inputs


@pytest.fixture(scope="module")
def made_up_run() -> ProbeRun:
    model, fit_inputs = made_up_model()
    return run_probes(model, fit_inputs, build_probes(fit_inputs))


def test_an_envelope_from_other_rows_is_noticed(made_up_run: ProbeRun) -> None:
    model, fit_inputs = made_up_model()
    assert is_envelope_rebuilt(model.envelope, fit_inputs)
    assert not is_envelope_rebuilt(model.envelope, fit_inputs[1:] * 1.001)
    only_one_maximum_moved = fit_inputs.copy()
    only_one_maximum_moved[np.argmax(fit_inputs[:, 0]), 0] += 1.0
    assert not is_envelope_rebuilt(model.envelope, only_one_maximum_moved)
    assert made_up_run.is_envelope_rebuilt


def test_end_to_end_the_typical_boundary_and_absurd_probes_behave(made_up_run: ProbeRun) -> None:
    # Made-up shaking is random, so whether its combination probes look unusual isn't tested here.
    for result in made_up_run.outcomes:
        if result.probe.kind != COMBINATION:
            assert is_as_expected(result), (result.probe.name, result.reasons)


def test_one_unexpected_probe_makes_the_run_a_miss(made_up_run: ProbeRun) -> None:
    passing = [result for result in made_up_run.outcomes if result.probe.kind != COMBINATION]
    assert has_passed(dataclasses.replace(made_up_run, outcomes=passing))
    wrongly_answered = dataclasses.replace(passing[-1], reasons=(), drift=ANSWER)
    assert not has_passed(dataclasses.replace(made_up_run, outcomes=[*passing[:-1], wrongly_answered]))
    assert not has_passed(dataclasses.replace(made_up_run, outcomes=passing, is_envelope_rebuilt=False))


def test_the_report_gives_the_verdict_and_marks_each_miss(made_up_run: ProbeRun) -> None:
    typical = next(result for result in made_up_run.outcomes if result.probe.kind == TYPICAL)
    refused_typical = dataclasses.replace(typical, reasons=("storeys | bays odd",), drift=None)
    report = format_probe_report(dataclasses.replace(made_up_run, outcomes=[refused_typical]), run_minutes=0.5)
    assert report.startswith("# Slice 3 refusal probes: MISS")
    assert "| typical | the typical fit row | **NO** | refused: storeys \\| bays odd |" in report
    assert np.isfinite(made_up_run.distance_limit)
