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

The reproduction, step by step. Downloading and fitting the records waits for approval
([docs/spikes.md](docs/spikes.md)), so `FITTED.json` doesn't exist yet:

```bash
uv run earthquake-screening select-records          # ESM metadata -> data/esm_selection.csv
uv run earthquake-screening build-pool FITTED.json  # synthetic records, simulated and labelled
uv run earthquake-screening reproduce               # the 20-start experiment vs the paper's Table 2
```

## Layout

| Path | What it does |
|---|---|
| `earthquake_screening/simulation/` | Everything that makes data: choosing ESM records, fitting and generating ground motions, the oscillator in OpenSees, the labelled pool |
| `earthquake_screening/surrogate/` | Active learning with a linear SVM, and fragility curves with their error |
| `earthquake_screening/reproduction/` | The Sainct et al. experiment, its pass rule and its report |
| `earthquake_screening/cli.py` | The commands above |
| `tests/` | One behaviour per test, named as a specification |
| `typings/` | Type stubs for OpenSeesPy and the parts of scikit-learn we call |
| `data/esm_selection.csv` | The ESM records chosen with the paper's filter (metadata only) |
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
