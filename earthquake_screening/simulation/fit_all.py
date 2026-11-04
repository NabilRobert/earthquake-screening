"""Fit every selected component in parallel; a record the model can't describe is listed, not patched."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from earthquake_screening.simulation.esm import SelectedComponent
from earthquake_screening.simulation.esm_records import read_component
from earthquake_screening.simulation.record_fit import RecordFitError, fit_record
from earthquake_screening.simulation.synthesis import FittedRecord


@dataclass(frozen=True)
class FitOutcome:
    record_id: str
    fitted: FittedRecord | None
    failure: str | None  # why the record couldn't be fitted


def _fit_one(task: tuple[SelectedComponent, Path, int]) -> FitOutcome:
    selected, records_dir, seed = task
    component = read_component(selected, records_dir)
    try:
        fit = fit_record(component.acceleration, component.time_step, seed)
    except RecordFitError as error:
        return FitOutcome(component.record_id, None, str(error))
    # Synthetic records start at the shaking, so they take the record's length after T0.
    duration = component.duration_s - fit.start_delay_s
    return FitOutcome(component.record_id, FittedRecord(component.record_id, fit.parameters, duration), None)


def fit_components(selected: list[SelectedComponent], records_dir: Path, seed: int) -> list[FitOutcome]:
    """One outcome per component, in the selection's order; each fit has its own seed."""
    tasks = [(component, records_dir, seed + index) for index, component in enumerate(selected)]
    with ProcessPoolExecutor() as executor:
        return list(executor.map(_fit_one, tasks))
