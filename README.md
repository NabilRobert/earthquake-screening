# earthquake-screening

Instant earthquake screening for buildings: a fast surrogate model trained on
nonlinear earthquake simulations. **Research and portfolio work only: it is not a
structural assessment, and an engineer signs off any real decision.**

The first slice reproduces a published result before claiming anything new:
Sainct et al. (2020), fragility curves for a nonlinear oscillator by active learning
on support vector machines. The pass rule was fixed before running:
[docs/reproduction/sainct2020.md](docs/reproduction/sainct2020.md).

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

## Layout

| Path | What it does |
|---|---|
| `earthquake_screening/simulation/` | Everything that makes data: choosing ESM records, fitting and generating ground motions, the oscillator in OpenSees, the labelled pool |
| `earthquake_screening/surrogate/` | Active learning with a linear SVM, and fragility curves with their error |
| `earthquake_screening/reproduction/` | The Sainct et al. experiment, its pass rule and its report |
| `earthquake_screening/cli.py` | The commands above |
| `tests/` | One behaviour per test, named as a specification |
| `scripts/` | Post-hoc diagnostics of the reproduction |
| `typings/` | Type stubs for OpenSeesPy and the parts of scikit-learn we call |
| `data/esm_selection.csv`, `data/esm_records.sha256`, `data/ESM_ATTRIBUTION.md` | Which ESM records, their pinned checksums, and the credit each file's network asks for |
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
  metadata.
- **Method:** Sainct R., Feau C., Martinez J.-M., Garnier J. (2020), Structural Safety
  86:101972; and Rezaeian S., Der Kiureghian A. (2008, 2010) for the ground-motion
  model.
