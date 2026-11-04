"""Command line: the reproduction's steps, each repeatable from its inputs and a seed.

select-records  ESM metadata -> data/esm_selection.csv (committed)
fetch-records   the selected recordings -> data/records/ (not committed), checksums pinned
fit-records     8 parameters per component -> data/generated/fitted.json, plus the attribution file
build-pool      fitted records -> synthetic records simulated and labelled (data/generated/, not committed)
reproduce       pool -> the 20-start experiment and the report against the paper's Table 2
fetch-prj2048   slice 2's part of DesignSafe PRJ-2048 -> data/prj2048/ (not committed), checksums pinned
train-buildings PRJ-2048 -> the building surrogate (data/generated/) and its judged result
probe-refusals  slice 2's saved model -> the refusal probe suite, judged by slice 3's fixed rule
"""

import time
from pathlib import Path

import typer

from earthquake_screening import prj2048
from earthquake_screening.buildings import probe_run
from earthquake_screening.buildings import report as building_report
from earthquake_screening.buildings.dataset import load_prj2048
from earthquake_screening.buildings.experiment import run_experiment
from earthquake_screening.buildings.probe_report import format_probe_report
from earthquake_screening.buildings.split import rows_for, split_prj2048
from earthquake_screening.reproduction.report import format_report
from earthquake_screening.reproduction.sainct2020 import run_reproduction
from earthquake_screening.screening.probes import build_probes
from earthquake_screening.simulation import esm, esm_records
from earthquake_screening.simulation.fit_all import fit_components
from earthquake_screening.simulation.oscillator import Oscillator
from earthquake_screening.simulation.pool import Pool, build_pool
from earthquake_screening.simulation.synthesis import draw_synthetic_specs, load_fitted, save_fitted

SELECTION = Path("data/esm_selection.csv")
RECORDS_DIR = Path("data/records")
CHECKSUMS = Path("data/esm_records.sha256")
# The paper's pool: 100,000 synthetic records before the window on L (section 2.3).
PAPER_SYNTHETIC_COUNT = 100_000

app = typer.Typer(no_args_is_help=True)


@app.command()
def select_records(output: Path = SELECTION) -> None:
    """Apply the paper's filter to ESM's metadata and pin the chosen components."""
    components = esm.select_components(esm.fetch_flatfile())
    esm.write_selection(components, str(output))
    recordings = len({component.record_id.rsplit(".", 1)[0] for component in components})
    typer.echo(f"{len(components)} components from {recordings} recordings -> {output}")


@app.command()
def fetch_records() -> None:
    """Download the selected recordings (or check the ones already here) against the pinned checksums."""
    fetched = esm_records.fetch_recordings(esm.read_selection(str(SELECTION)), RECORDS_DIR, CHECKSUMS)
    typer.echo(f"fetched {fetched / 1e6:.1f} MB; checksums in {CHECKSUMS}")


@app.command()
def fit_records(output: Path = Path("data/generated/fitted.json"), seed: int = 2020) -> None:
    """Fit the 8 ground-motion parameters to every selected component."""
    selected = esm.read_selection(str(SELECTION))
    components = [esm_records.read_component(component, RECORDS_DIR) for component in selected]
    Path("data/ESM_ATTRIBUTION.md").write_text(esm_records.attribution_markdown(components))
    outcomes = fit_components(selected, RECORDS_DIR, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_fitted([outcome.fitted for outcome in outcomes if outcome.fitted is not None], output)
    for outcome in outcomes:
        if outcome.failure is not None:
            typer.echo(f"not fitted: {outcome.record_id}: {outcome.failure}")
    fitted_count = sum(outcome.fitted is not None for outcome in outcomes)
    typer.echo(f"{fitted_count} of {len(outcomes)} components fitted -> {output}")


@app.command()
def fetch_prj2048() -> None:
    """Download slice 2's part of PRJ-2048 (or check what's here) against the pinned checksums."""
    fetched = prj2048.fetch(Path("data/prj2048"), Path("data/prj2048.sha256"))
    Path("data/PRJ2048_ATTRIBUTION.md").write_text(prj2048.ATTRIBUTION)
    typer.echo(f"fetched {fetched / 1e6:.1f} MB; checksums in data/prj2048.sha256")


@app.command(name="build-pool")
def build_pool_from_fitted(
    fitted: Path,
    output: Path = Path("data/generated/pool.npz"),
    count: int = PAPER_SYNTHETIC_COUNT,
    seed: int = 2020,
) -> None:
    """Draw synthetic records around the fitted ones, simulate them, and label the ones in the window."""
    specs = draw_synthetic_specs(load_fitted(fitted), count, seed)
    pool = build_pool(specs, Oscillator())
    output.parent.mkdir(parents=True, exist_ok=True)
    pool.save(str(output))
    typer.echo(f"{int(pool.labelled_rows.sum())} labelled records of {count} -> {output}")


@app.command()
def reproduce(
    pool: Path = Path("data/generated/pool.npz"),
    output: Path = Path("docs/reproduction/sainct2020-result.md"),
    seed: int = 2020,
) -> None:
    """Run the 20-start experiment and write the comparison with the paper's Table 2."""
    loaded = Pool.load(str(pool))
    report = format_report(run_reproduction(loaded, seed), loaded, seed)
    output.write_text(report)
    typer.echo(report)


@app.command()
def train_buildings(
    output: Path = Path("docs/building-surrogate/prj2048-result.md"),
    model_path: Path = Path("data/generated/building_model.pkl"),
    seed: int = 2048,
) -> None:
    """Split PRJ-2048 by frame and ground motion, train the surrogate, and judge it by the fixed pass rule."""
    started = time.perf_counter()
    model, result = run_experiment(load_prj2048(Path("data/prj2048")), seed)
    model_sha256 = building_report.save_model(model, model_path)
    report = building_report.format_report(result, model_sha256, (time.perf_counter() - started) / 60)
    output.write_text(report)
    typer.echo(report)


@app.command()
def probe_refusals(
    output: Path = Path("docs/building-surrogate/refusal-probes-result.md"),
    model_path: Path = Path("data/generated/building_model.pkl"),
) -> None:
    """Put the probe suite to slice 2's saved model and judge it by slice 3's fixed rule."""
    started = time.perf_counter()
    model = probe_run.load_slice2_model(model_path)
    data = load_prj2048(Path("data/prj2048"))
    split = split_prj2048(data)
    fit_inputs = rows_for(data, split.fit_frames, split.training_motions).inputs
    run = probe_run.run_probes(model, fit_inputs, build_probes(fit_inputs))
    report = format_probe_report(run, (time.perf_counter() - started) / 60)
    output.write_text(report)
    typer.echo(report)
