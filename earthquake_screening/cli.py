"""Command line: the reproduction's steps, each repeatable from its inputs and a seed.

    select-records  ESM metadata -> data/esm_selection.csv (committed)
    build-pool      fitted records -> synthetic records simulated and labelled (data/generated/, not committed)
    reproduce       pool -> the 20-start experiment and the report against the paper's Table 2

Fitting the real records needs their waveforms, which wait for Arthur's OK to download.
"""

from pathlib import Path

import typer

from earthquake_screening.reproduction.report import format_report
from earthquake_screening.reproduction.sainct2020 import run_reproduction
from earthquake_screening.simulation import esm
from earthquake_screening.simulation.oscillator import Oscillator
from earthquake_screening.simulation.pool import Pool, build_pool
from earthquake_screening.simulation.synthesis import draw_synthetic_specs, load_fitted

# The paper's pool: 100,000 synthetic records before the window on L (section 2.3).
PAPER_SYNTHETIC_COUNT = 100_000

app = typer.Typer(no_args_is_help=True)


@app.command()
def select_records(output: Path = Path("data/esm_selection.csv")) -> None:
    """Apply the paper's filter to ESM's metadata and pin the chosen components."""
    components = esm.select_components(esm.fetch_flatfile())
    esm.write_selection(components, str(output))
    recordings = len({component.record_id.rsplit(".", 1)[0] for component in components})
    typer.echo(f"{len(components)} components from {recordings} recordings -> {output}")


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
