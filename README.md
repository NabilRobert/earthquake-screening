# earthquake-screening

Instant earthquake screening for buildings: a fast surrogate model trained on
nonlinear earthquake simulations. **Research and portfolio work only: it is not a
structural assessment, and an engineer signs off any real decision.**

The first slice reproduces a published result before claiming anything new:
Sainct et al. (2020), fragility curves for a nonlinear oscillator by active learning
on support vector machines. The pass rule was fixed before running:
[docs/reproduction/sainct2020.md](docs/reproduction/sainct2020.md).

The second slice trains a building-level surrogate on DesignSafe PRJ-2048 (621 steel
moment frames, each simulated under 240 ground motions). It predicts a frame's peak
inter-storey drift with a 90% interval, and refuses inputs outside its training
range. It is judged on frames and ground motions it never saw, against a
Hazus-style class baseline:
[docs/building-surrogate/prj2048.md](docs/building-surrogate/prj2048.md).

The third slice tests the refusals. 72 probes are put to the saved model: values
just inside and just outside each bound, impossible spectra, and absurd values
such as 0 storeys, NaN or "١٤". Every probe must be refused with a reason naming
its field, or answered, as the rule fixed beforehand says:
[docs/building-surrogate/refusal-probes.md](docs/building-surrogate/refusal-probes.md).

## Run it

```bash
uv sync
uv run pytest
```

The reproduction, step by step (about 1 h 45 min on an 8-core laptop, $0). It was run
once and **missed** the paper's numbers; see
[docs/reproduction/sainct2020.md](docs/reproduction/sainct2020.md#result-a-miss-run-once-2026-11-03).

```bash
uv run earthquake-screening select-records   # ESM metadata -> data/esm_selection.csv
uv run earthquake-screening fetch-records    # 16 MB of ESM records, checked against data/esm_records.sha256
uv run earthquake-screening fit-records      # 8 parameters per component -> data/generated/fitted.json
uv run earthquake-screening build-pool data/generated/fitted.json   # 100,000 synthetic records, labelled
uv run earthquake-screening reproduce        # the 20-start experiment vs the paper's Table 2
```

The building surrogate (about 6 min, $0). It was run once and **passed** its rule, with
the limits listed in [the result](docs/building-surrogate/prj2048.md#result-pass-run-once-2026-11-03):

```bash
uv run earthquake-screening fetch-prj2048    # 35 MB of PRJ-2048, checked against data/prj2048.sha256
uv run earthquake-screening train-buildings  # split, train, calibrate, judge -> docs/building-surrogate/prj2048-result.md
```

The refusal probes (a few seconds, $0; needs the model from `train-buildings`). They
were run once and **passed**; see
[the result](docs/building-surrogate/refusal-probes.md#result-pass-run-once-2026-11-04):

```bash
uv run earthquake-screening probe-refusals   # 72 probes vs slice 2's model -> docs/building-surrogate/refusal-probes-result.md
```

## Layout

| Path | What it does |
|---|---|
| `earthquake_screening/simulation/` | Everything that makes data: choosing ESM records, fitting and generating ground motions, the oscillator in OpenSees, the labelled pool |
| `earthquake_screening/surrogate/` | Active learning with a linear SVM, and fragility curves with their error |
| `earthquake_screening/reproduction/` | The Sainct et al. experiment, its pass rule and its report |
| `earthquake_screening/prj2048.py` | Fetching slice 2's part of PRJ-2048, with pinned checksums |
| `earthquake_screening/buildings/` | Slices 2 and 3: reading PRJ-2048, the split by frame and ground motion, the runs and their reports |
| `earthquake_screening/surrogate/building_model.py`, `class_baseline.py` | The two candidate surrogates and the Hazus-style baseline |
| `earthquake_screening/screening/` | What a screen runs: checking the request, the training envelope that refuses unfamiliar inputs, and the probes that test both. Never imports OpenSeesPy |
| `earthquake_screening/cli.py` | The commands above |
| `tests/` | One behaviour per test, named as a specification |
| `scripts/` | Post-hoc diagnostics of the reproduction |
| `typings/` | Type stubs for OpenSeesPy, MAPIE and the parts of scikit-learn we call |
| `data/esm_selection.csv`, `data/esm_records.sha256`, `data/ESM_ATTRIBUTION.md` | Which ESM records, their pinned checksums, and the credit each file's network asks for |
| `data/prj2048.sha256`, `data/PRJ2048_ATTRIBUTION.md` | PRJ-2048's pinned checksums and its credit |
| `docs/guide.md` | A walk through one run, a glossary and where to change things |

Records and generated data live in `data/` and are never committed.

## Licences and credits

- This repository's code: MIT ([LICENSE](LICENSE)).
- **OpenSeesPy** has its own terms: free for research, education and internal use; a
  commercial application or cloud service that imports it needs a licence from UC
  Berkeley. Only `earthquake_screening/simulation/` imports it, offline, to make data.
- **ESM** (Engineering Strong Motion database, CC BY 4.0): Mascandola C., Felicetta C.,
  Russo E., Luzi L., Lanzano G., Sgobba S., Brunelli G., Ramadan F., Pacor F., ORFEUS
  Strong-Motion SMC (2026). Engineering Strong Motion Database (ESM), version 3.0.
  INGV. https://doi.org/10.13127/esm.3. `data/esm_selection.csv` is derived from its
  metadata. Each file states its own licence: 64 of the 148 components say "unknown",
  so the records, and the data generated from them, are never redistributed or
  published; only their list and checksums are in the repo.
- **PRJ-2048** (Open Data Commons Attribution): Guan X., Burton H., Shokrabadi M.
  (2020). A Database of Seismic Designs, Nonlinear Models, and Seismic Responses for
  Steel Moment Resisting Frame Buildings. DesignSafe-CI.
  https://doi.org/10.17603/ds2-8yc7-1285. Used here, not redistributed.
- **Method:** Sainct R., Feau C., Martinez J.-M., Garnier J. (2020), Structural Safety
  86:101972; and Rezaeian S., Der Kiureghian A. (2008, 2010) for the ground-motion
  model.
