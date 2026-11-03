# Slice Plan

## Prerequisites (outside this repo)

| # | Prerequisite | Needed for | Who |
|---|---|---|---|
| P1 | Arthur OKs Phases 1, 2 and 5 | Slice 1 | Arthur |
| P2 | The GitHub repo exists | Pushing anything | Arthur |
| P3 | OK to download the 97 ESM records | Slice 1 | Arthur |
| P4 | OK to download PRJ-2048's response files | Slice 2 | Arthur |
| P5 | The friend agrees to review each slice's set-up | Every slice | Arthur |
| P6 | OK for any real-model eval run | Slice 5 | Arthur |

## Spikes (local, $0)

| # | Question | Method | Pass | Time |
|---|---|---|---|---|
| 1 | Can we get the ESM records, and on what terms? | Read ESM's access page and the dataset record; list the 97 records' size | Terms written into sources.md; size known | 30 min |
| 2 | What's in PRJ-2048, and how big is it? | Read its file listing on DesignSafe | Responses can be fetched without the models; size known | 30 min |
| 3 | Does OpenSeesPy install and run fast enough? | Install it in a venv on macOS arm64 and in Linux CI; time 100 oscillator runs | Both install; time per run known | 1 h |
| 4 | Is there a maintained Rezaeian generator? | Search PyPI and SimCenter | Use it, or confirm we write it | 30 min |

## Slices

The tests and target numbers come first. Each new test is mutation-checked
once. Each slice updates `docs/guide.md`, and the friend checks its set-up.

| Slice | Delivers | Done when | Cost |
|---|---|---|---|
| **1. Reproduce Sainct 2020** | Scaffold, CI, complexity rules; the record generator; the oscillator in OpenSeesPy; the SVM active-learning loop; the reproduction report | See below | $0 |
| **2. Building surrogate** | PRJ-2048 loader; split by building; gradient boosting vs GP; the Hazus class baseline from pelicun; MAPIE intervals | Held-out error and coverage reported against the baseline (G2, G3); a test proves no frame is in both splits | $0 |
| **3. Refusals** | Envelope saved with the model; refusal with reasons; probe suite | G4: every out-of-range probe refused, none inside | $0 |
| **4. Screen and report** | Drift to damage states via pelicun; `screen` Typer command; report with the notice; the no-`openseespy` import test | Under 1 s on a laptop (G5); every output carries the notice | $0 |
| **5. Agent layer** | PydanticAI agent with one tool; Inspect AI eval on a recorded model | G7: no number in a report that isn't in the tool's output | $0 in CI; any real run needs Arthur's OK |
| **6. Evidence and write-up** | RESULTS from scripts; JOURNEY; limits; portfolio entry (with the portfolio session) | Every number traces to a file; the licence notes are in the README | $0 |

**After v1, only if Arthur decides to sell:** letters to UC Berkeley and
CSI, a licensed engineer, RC frames for Indonesia and Turkey, and the
ETABS add-on (ADR-004).

## Slice 1 in detail

**Scope:** one oscillator, synthetic records, SVM. No buildings yet.

**Delivers:**
- `pyproject.toml` (uv), ruff with the complexity rules, pyright strict;
  `.github/workflows/ci.yml`; a README with a layout section and the
  OpenSeesPy licence note; `docs/guide.md`;
- `simulation/records.py`: fetch the 97 ESM records, fit the Rezaeian
  model, draw synthetic records with a seed;
- `simulation/oscillator.py`: the 5 Hz oscillator in OpenSeesPy, returning
  peak displacement;
- `surrogate/active_svm.py`: the paper's active-learning loop;
- `reproduction/sainct2020.py`: 20 runs at 100 and 1,000 simulations, and a
  report with ΔL2 against Table 2.

**Order:**
1. Write the pass rule and a test for the ΔL2 formula (Eq. 37) on a
   hand-worked case. Tests fail.
2. Oscillator: check it against a hand calculation in the elastic range,
   with the friend.
3. Records: fit, then check the synthetic records' statistics against the
   fitted ones.
4. Reference curve from all the records (about 33,700), then the SVM loop.
5. Spend a pass trying to break it: zero-length records, a yield of 0,
   NaN in the parameters, a seed that changes the answer.

**Done when:**
- the mean ΔL2 over 20 runs is inside 2.6% ± 1 (100 simulations) and
  1.6% ± 0.3 (1,000), or the miss is reported honestly with its cause;
- the run is repeatable from one command and a seed;
- CI runs a small subset in under 5 minutes, and the new tests are
  mutation-checked;
- the generated data and records are not in git.
