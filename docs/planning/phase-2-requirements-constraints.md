# Phase 2: Requirements & Constraints

Phases 3 (threats) and 4 (design) are folded into this doc and Phase 5, as
in Projects 7 and 9. The build goes in slices ([slice plan](slice-plan.md)).

## Core scenario (end of v1)

1. **Describe.** "8-storey reinforced concrete frame, built 1995, Jakarta,
   storey height 3.5 m." The agent turns this into a typed
   `BuildingDescription`. If a required field is missing, it asks. It never
   guesses a structural value.
2. **Check the range.** Code compares the description with the training
   envelope. Outside it, the answer is "out of range: ask an engineer", with
   the reason (for example, "16 storeys; trained on 2 to 12").
3. **Screen.** The surrogate returns, for the given shaking intensity, the
   peak inter-storey drift with an interval, and the probability of each
   damage state.
4. **Report.** A short report quotes those numbers, the model's measured
   held-out error, the intensity used, and the screening-only notice. Every
   number comes from the surrogate's output (G7).

## Requirements

**Simulation and data (the slow part, $0)**
- **S1.** Each simulated building is generated from a parameter set by a
  script, with a fixed seed. The same seed gives the same numbers.
- **S2.** Each run records its inputs, the engine and version, the
  ground-motion record IDs, the outputs and the run time.
- **S3.** Failed or non-converging runs are kept and counted, never dropped
  silently. A collapse is a result, not an error.
- **S4.** Ground-motion records aren't committed (their terms decide, see
  [sources.md](sources.md)). A script fetches or reads them from a local
  folder and checks their hashes.
- **S5.** Generated data isn't committed. One summary file and the script
  that regenerates it are.

**The surrogate**
- **M1.** Split by **building**, not by row: no held-out building's runs
  appear in training, under any ground motion.
- **M2.** Report the error on the held-out buildings, per output, with the
  baseline (Hazus-style class table) on the same buildings.
- **M3.** Every estimate has an interval. Its coverage is measured on the
  held-out set and reported next to the nominal rate.
- **M4.** The model can run on a laptop CPU, with no GPU, in under 1 s.
- **M5.** The training script is deterministic given its seed. The trained
  model's file hash is recorded with its metrics.

**Refusing outside the range**
- **R1.** The training envelope (each input's range, plus a check on
  combinations) is computed from the training data and stored with the
  model.
- **R2.** Inputs outside it are refused with the reason. There is no
  "low confidence" number instead.
- **R3.** A probe suite tests just inside and just outside each boundary,
  plus absurd values (0 storeys, negative heights, unicode digits, NaN).

**The agent layer (slice 5)**
- **A1.** The model sees the user's description and returns a typed
  `BuildingDescription`. It has one tool, `screen_building`, and no other
  way to produce numbers.
- **A2.** Missing or ambiguous structural fields mean asking. Code
  validates every field's units and range.
- **A3.** The report's numbers are filled in by code from the tool's
  output. The model writes only the text around them.
- **A4.** Evals run on Inspect AI, with a free local or recorded model, at
  $0. Any paid run needs Arthur's OK.

**Liability**
- **L1.** Every output, in the CLI, the API and the report, carries the
  screening-only notice: it isn't a structural assessment, and an engineer
  signs off any decision.
- **L2.** The README and the guide state the model's scope, its measured
  error and what it doesn't cover.
- **L3.** No output calls a building "safe".

## Constraints

| Constraint | Limit | Why |
|---|---|---|
| Money | $0: no paid compute, models or data | Arthur's rule |
| Compute | A laptop CPU, or the small shared server off-peak | Nothing else is free |
| Downloads | Nothing large without Arthur's OK | Arthur's rule |
| Licences | No runtime `import openseespy`; no OpenQuake import in v1; no ETABS | [Licence gate](licence-gate.md) |
| Data | No third-party records or models committed | Their terms; S4 |
| Reviewer | A civil-engineering student checks the models; no one signs off | ADR-010 |
| Stack | Python, uv, Pydantic, pytest, ruff, pyright; FastAPI only if a service is needed; Inspect AI for evals | Folder rules |

## Threats (folded Phase 3)

| Threat | Effect | Mitigation |
|---|---|---|
| Someone treats a screen as an assessment | Real harm and liability | L1–L3; no hosted service in v1 |
| The model extrapolates quietly | A wrong "low" answer | R1–R3 |
| Leakage between training and test | Error looks better than it is | M1; a test that checks the split |
| The reproduction is tuned until it matches | A false proof | Tolerance fixed before running; every attempt logged |
| The agent invents a number | A wrong report | A1, A3, G7 |
| A licence is breached | The project can't be shown or sold | The licence gate; no records or ETABS files in the repo |
| Simulation time runs away | No data at $0 | Time budget per slice; a small model type first |
