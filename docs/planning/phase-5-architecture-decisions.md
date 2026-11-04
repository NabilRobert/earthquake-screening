# Phase 5: Architecture Decisions

Each decision lists the options considered, as the folder rules ask.
"Settled" means Arthur agreed on 2026-11-03. Everything else is a
recommendation until he does.

## Reuse map (the folder's "reuse before building" rule)

**Arthur's repos:** none has ML, simulation or seismic code (grep for
scikit-learn, scipy, torch and similar across the folder found only the
vendored Inspect AI checkout). What carries over:

| Need | Reuse | When |
|---|---|---|
| Agent eval | Inspect AI, the pattern from `agent-eval-harness` | Slice 5 |
| Repo layout, CI, complexity rules, guide | Project 9's slice 1 (pyproject, ruff rules, pyright strict) | Slice 1 |
| Exposing the screen as a tool later | MCP-backbone | After v1 |

**Everyone else's:**

| Need | Use | Licence | Instead of |
|---|---|---|---|
| Nonlinear simulation | OpenSeesPy (offline only) | UC custom, internal use | Writing a solver |
| Surrogates, SVMs | scikit-learn | BSD-3 | PyTorch |
| Intervals with coverage | MAPIE 1.6 (2026-10-19) | BSD-3 | Hand-written conformal code |
| Damage states from drift | pelicun 3.10 and its Hazus/P-58 library (DLML) | BSD-3 | Typing in Hazus tables |
| Building-level training data | DesignSafe PRJ-2048, 621 steel frames | ODC-By | Simulating 621 frames ourselves |
| Real records | ESM v2.0 | CC BY 4.0 | NGA-West2 (ADR-005) |

**Built here, and why:** the Rezaeian–Der Kiureghian ground-motion
generator (about 100 lines; it is part of what slice 1 reproduces, and
SimCenter's `smelt` is a C++ build needing Intel MKL and IPP), the
active-learning loop (it is the paper's method), the split by building, the
range check and the reports. Those are what the project demonstrates.

## ADR-001: Portfolio and research first; selling is later (settled)

**Decision:** an open repo under Arthur's usual MIT licence, built as a
research and portfolio piece at $0. The commercial path (ADR-004) is
written down but not built.

**Why:** OpenSeesPy is free for internal and research use only, and the
questions that decide a sale (outputs, "internal purposes", ETABS results)
need UC Berkeley, CSI and a lawyer ([licence gate](licence-gate.md)). None
of that blocks a portfolio.

**Rejected:** *aiming to sell now:* every slice would wait on letters to
Berkeley and CSI.

## ADR-002: OpenSeesPy runs offline only; nothing at runtime imports it (settled)

**Decision:** simulation scripts live in their own package
(`simulation/`), run by hand or by CLI to make data. The screening code
(`screening/`) loads only a trained model file. A test fails if anything
under `screening/` imports `openseespy`.

**Why:** OpenSeesPy's own licence note names a "cloud-based service that
uses `import openseespy`" as needing a commercial licence. Keeping that
line now means the commercial path doesn't need a rewrite.

**Rejected:**
- *Writing our own Newmark solver:* fine for a single oscillator, but
  slice 1's results must be comparable to the field's tool, and later
  frames need it anyway.
- *Code_Aster or Kratos:* much heavier, and not what the papers use.

## ADR-003: OpenQuake is not in v1

**Decision:** the shaking intensity is an input. No OpenQuake import, no
GEM hazard data.

**Why:** importing `openquake.hazardlib` binds our code to the AGPL and
would mix it with OpenSees's non-commercial terms. GEM's global hazard map
is CC BY-NC-SA. v1 doesn't need site hazard to prove the surrogate.

**Later:** run the engine as a separate program for hazard inputs, and
check each GEM product's licence.

## ADR-004: ETABS is the commercial path, recorded and not built

**Decision (for later):**
- Train on an open engine.
- An ETABS add-on runs **on the customer's machine, under their licence**,
  reads their model through CSI's API and sends the building's description
  to the screen.
- Validate against ETABS results that a licensed partner engineer runs
  under their own licence, on models they own.
- **Never** host ETABS for others: the EULA's §4(b) and §4(d) bar it.

**Not now, because:** Arthur has no licence, a trial bans "commercial or
professional use", and the friend's academic copy is for teaching and
research only. Three questions for CSI are in the licence gate.

**Rejected:** *training on ETABS outputs:* needs our own commercial
licence, and the EULA is silent on outputs. *Any unlicensed copy:* never.

## ADR-005: Ground motions: ESM and synthetic records; no NGA-West2 in v1

**Decision:**
- Slice 1 uses the paper's own route: 97 ESM records (CC BY 4.0), fitted
  to the Rezaeian–Der Kiureghian model, then synthetic records drawn from
  the fit.
- Records are fetched by a script into `data/records/` (ignored by git),
  with hashes checked.
- Slice 2 uses PRJ-2048's precomputed responses (ODC-By, attributed). We
  use its drift numbers, not its waveforms, and never redistribute it.

**Why not NGA-West2:** its terms are "All Rights Reserved", with no grant
to redistribute or for commercial use. Accounts need a university or
company email, last one year, and downloads are capped "due to copyright
issues". K-NET is barred from redistribution. AFAD and BMKG terms couldn't
be confirmed.

## ADR-006: Reproduce Sainct et al. (2020) first (settled)

**Decision:** slice 1 reproduces Sainct, Feau, Martinez and Garnier,
*Efficient seismic fragility curve estimation by active learning on
support vector machines* (Structural Safety 86, 2020;
[arXiv:1810.01240](https://arxiv.org/abs/1810.01240)).
- Model: a nonlinear single-mass oscillator, 5 Hz, 2% damping, yield at
  5 mm, post-yield stiffness 20%.
- Failure: peak displacement over twice the yield.
- Target (their Table 2, PGA, 4-D linear SVM, mean of 20 runs): fragility
  error ΔL2 of **2.6% ± 1 at 100 simulations and 1.6% ± 0.3 at 1,000**.

**Pass rule, fixed before running:** our mean ΔL2 over 20 runs falls inside
their band at both sizes. Every attempt is logged. A miss is reported as a
miss, with what differed.

**Known gap:** the paper says the record model has 9 parameters in one
place and 8 in another. We pick one, say which, and note it.

**Then:** slice 2 moves to building-level drift on PRJ-2048, which is
closer to the product.

**Rejected:**
- *PRJ-2048 alone:* the paper is paywalled and its headline numbers
  couldn't be confirmed, so the "reproduction" would be weak.
- *Li and Ceferino (2026):* a preprint, and further from buildings. It is
  kept as a reference for where neural surrogates fail.

## ADR-007: The surrogate is scikit-learn, split by building (settled framework)

**Decision:**
- Slice 2 inputs: frame features from PRJ-2048 (storeys, period, design
  parameters) and the shaking intensity. Output: peak inter-storey drift.
- Candidates: gradient boosting and Gaussian processes. The pick is the
  better one on the dev split.
- **Split by building:** 20% of frames held out entirely.
- **Baseline:** the Hazus steel moment-frame class, from pelicun's
  library, on the same held-out frames.

**Rejected:** *PyTorch or neural operators:* small tabular data on a CPU,
and Li and Ceferino (2026) report large errors for nonlinear responses.
*A random row split:* leaks each building into its own test set.

## ADR-008: Intervals with measured coverage (settled framework)

**Decision:** MAPIE's conformal intervals, at 90%, calibrated on a part of
the training buildings. Coverage on held-out buildings is reported next to
90%. A miss of more than 2 points fails the check.

**Rejected:** *quantile regression alone:* no coverage guarantee.
*Hand-written conformal code:* MAPIE is maintained and BSD-3.

## ADR-009: Refuse outside the training range

**Decision:**
- The envelope is saved with the model: each input's training min and max,
  plus a nearest-neighbour distance limit, set so 99% of held-out
  in-range buildings pass.
- Outside it, the answer is a refusal with the reason, never a number.
- Probe suite: just inside and just outside each bound, combinations never
  seen together, and absurd values.

**Rejected:** *returning a wider interval instead:* readers see a number
and use it.

## ADR-010: Reviewer and liability

**Decision:**
- Arthur's friend, a civil-engineering student, checks the model set-up
  (units, the oscillator, the frame features) each slice. **No one signs
  anything off.** The README says so.
- Every output carries the notice: screening only, not a structural
  assessment, an engineer signs off any decision. No output says "safe".
- Before any claim beyond research (a pilot, a customer, a sale), a
  licensed structural engineer reviews the method. Ideally they hold an
  ETABS licence (ADR-004).

**Rejected:** *presenting the student's review as engineering sign-off.*

## ADR-011: Compute: a laptop at $0

**Decision:** everything runs on Arthur's laptop CPU. Slice 1 is about
33,700 short oscillator runs, roughly an hour (an estimate checked in spike
3). CI runs a small fixed subset only. The shared server is used off-peak,
only with Arthur's OK.

**Rejected:** *cloud credits or GPUs:* not $0, and not needed.

## ADR-012: The agent layer is small, and it never makes numbers

**Decision:**
- A PydanticAI agent (Arthur's stack) with one tool, `screen_building`.
- Code fills the report's numbers from the tool's output. The model writes
  only the words around them.
- Evals on Inspect AI: a recorded or mock model in CI at $0. A real-model
  run needs Arthur's OK (spend rule).

**Rejected:** *the model reading values from the text and writing them into
the report:* the same failure Project 9 guards against.

## Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Portfolio first; selling later (ADR-001) | Settled |
| 2 | Reviewer: a civil-engineering student, with no sign-off (ADR-010) | Settled |
| 3 | Reproduce Sainct 2020, then PRJ-2048 (ADR-006) | Settled |
| 4 | New frameworks: OpenSeesPy, scikit-learn, MAPIE | Settled |
| 5 | Folder name `earthquake-screening` | Kept; no reason to change |
| 6 | Everything else above | Recommended |

## The first slice

The reproduction of Sainct et al. (2020). See the
[slice plan](slice-plan.md#slice-1-in-detail).

## Open questions

**For Arthur:**
1. Will your friend review each slice's model set-up? An hour a slice.
2. Downloads: the 97 ESM records (size checked in spike 1) and PRJ-2048
   (size checked in spike 2) each need your OK.
3. Do you know a licensed structural engineer, now or later? Not needed
   for v1.

**For the vendors (only before a sale):** see the
[licence gate](licence-gate.md#who-to-ask-when-it-matters).

**For the spikes:** ESM's registration and terms of access; PRJ-2048's
size and layout; OpenSeesPy wheels for macOS arm64 and Linux CI; run time
per oscillator run.

## Sources

The [licence gate](licence-gate.md) and [sources](sources.md), all read
2026-11-03.
