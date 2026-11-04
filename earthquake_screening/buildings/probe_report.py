"""Write slice 3's probe result as Markdown: the verdict, the checks, and every probe with its answer."""

from earthquake_screening.buildings.probe_run import (
    SLICE2_MODEL_SHA256,
    ProbeOutcome,
    ProbeRun,
    has_passed,
    is_as_expected,
)
from earthquake_screening.screening.probes import ABSURD, COMBINATION, JUST_INSIDE, JUST_OUTSIDE, TYPICAL

KIND_ORDER = (TYPICAL, JUST_INSIDE, JUST_OUTSIDE, COMBINATION, ABSURD)


def _answer(outcome: ProbeOutcome) -> str:
    if outcome.drift is None:
        return "refused: " + "; ".join(outcome.reasons)
    predicted, low, high = outcome.drift
    return f"answered: drift {predicted:.4f} (90% interval {low:.4f} to {high:.4f})"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _kind_rows(run: ProbeRun) -> list[str]:
    rows = ["| Kind | Probes | Expected | As expected |", "|---|---|---|---|"]
    for kind in KIND_ORDER:
        outcomes = [outcome for outcome in run.outcomes if outcome.probe.kind == kind]
        if not outcomes:
            continue
        expected = "refused" if outcomes[0].probe.should_refuse else "answered"
        matched = sum(is_as_expected(outcome) for outcome in outcomes)
        rows.append(f"| {kind} | {len(outcomes)} | {expected} | {matched} of {len(outcomes)} |")
    return rows


def _probe_rows(run: ProbeRun) -> list[str]:
    rows = ["| Kind | Probe | As expected | Answer |", "|---|---|---|---|"]
    for outcome in run.outcomes:
        mark = "yes" if is_as_expected(outcome) else "**NO**"
        rows.append(f"| {outcome.probe.kind} | {_cell(outcome.probe.name)} | {mark} | {_cell(_answer(outcome))} |")
    return rows


def format_probe_report(run: ProbeRun, run_minutes: float) -> str:
    verdict = "PASS" if has_passed(run) else "MISS"
    distances = [f"  - {name}: {distance:.2f}" for name, distance in run.combination_distances.items()]
    rebuilt = "yes" if run.is_envelope_rebuilt else "**no**"
    return "\n".join(
        [
            f"# Slice 3 refusal probes: {verdict}",
            "",
            "Judged by the rule fixed in [refusal-probes.md](refusal-probes.md).",
            "",
            f"- Model: slice 2's, SHA-256 `{SLICE2_MODEL_SHA256}` (checked on loading).",
            f"- Fit rows rebuilt from PRJ-2048 give the saved envelope's ranges: {rebuilt}.",
            f"- Envelope distance limit: {run.distance_limit:.2f}. The combination probes' distances:",
            *distances,
            f"- Run time: {run_minutes:.1f} min on the laptop, $0.",
            "",
            *_kind_rows(run),
            "",
            "## Every probe",
            "",
            *_probe_rows(run),
            "",
        ]
    )
