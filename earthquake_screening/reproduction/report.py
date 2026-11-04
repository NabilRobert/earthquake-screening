"""Write the reproduction's result as a Markdown table next to the paper's numbers."""

import numpy as np

from earthquake_screening.reproduction.sainct2020 import (
    FRAGILITY_GROUP_COUNT,
    JUDGED_PROJECTION,
    PAPER_TABLE_2,
    RowResult,
    has_passed,
)
from earthquake_screening.simulation.pool import Pool


def pool_summary(pool: Pool) -> list[str]:
    labelled = pool.labelled_rows
    not_converged = int((pool.is_in_window & ~pool.did_converge).sum())
    failure_rate = float(np.mean(pool.has_failed[labelled])) if labelled.any() else float("nan")
    return [
        f"- Synthetic records simulated: {pool.is_in_window.size}",
        f"- In the window Y <= L <= 6Y: {int(pool.is_in_window.sum())}",
        f"- OpenSees didn't converge (kept, counted, not labelled): {not_converged}",
        f"- Labelled pool: {int(labelled.sum())}, of which {failure_rate:.1%} failed",
    ]


def format_report(rows: list[RowResult], pool: Pool, seed: int) -> str:
    verdict = "PASS" if has_passed(rows) else "MISS"
    lines = [
        "# Sainct et al. (2020), Table 2: reproduction result",
        "",
        f"Seed {seed}; {len(rows[0].errors_percent)} test cases; {FRAGILITY_GROUP_COUNT} k-means groups.",
        f"Judged on the `{JUDGED_PROJECTION}` rows: **{verdict}**.",
        "",
        *pool_summary(pool),
        "",
        "| Curve against | Labelled records | Ours, Delta L2 % | Paper, Delta L2 % | Inside the paper's band |",
        "|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda row: (row.projection, row.budget)):
        paper_mean, paper_deviation = PAPER_TABLE_2[(row.projection, row.budget)]
        lines.append(
            f"| {row.projection} | {row.budget} | {row.mean:.2f} +/- {row.standard_deviation:.2f} "
            f"| {paper_mean} +/- {paper_deviation} | {'yes' if row.is_inside_paper_band else 'no'} |"
        )
    return "\n".join(lines) + "\n"
