# Spikes for slice 1 (2026-11-03)

From the slice plan (`docs/planning/slice-plan.md` on the `docs/planning` branch).
Each one is measured, with its method, so it can be checked again.

## 1. The ESM records: terms and size

- **Terms:** ESM 3.0 is CC BY 4.0 (DataCite record for doi:10.13127/esm.3, read
  2026-11-03). Cite Mascandola et al. (2026), ESM version 3.0, INGV,
  https://doi.org/10.13127/esm.3. Waveforms are public: the event-data web service
  needs no account (a token is only for embargoed data).
- **Selection:** the paper's filter on ESM's metadata service (one CSV, about 1 MB)
  gives 148 horizontal components from 74 recordings and 18 events
  ([deviations](reproduction/sainct2020.md#known-differences-from-the-paper)).
- **Size, measured on download (2026-11-03):** 74 recordings, 16.0 MB of zips, checksums
  in `data/esm_records.sha256`. Estimated beforehand at 12 to 46 MB.
- **Licences inside the files:** each file states its own: 64 components "U (unknown
  license)", 46 CC BY 3.0 IT, 36 CC BY 4.0, 2 "network default". They are used here,
  not redistributed, and every network is credited in `data/ESM_ATTRIBUTION.md`.

## 2. DesignSafe PRJ-2048 (slice 2): what slice 2 needs, and its size

Measured from DesignSafe's file listings (no files downloaded):

| Folder | Files | Size | Slice 2 needs it? |
|---|---|---|---|
| Top-level CSVs (building IDs, design summary, read-me) | 4 | 30 KB | Yes |
| `StructuralResponses/EDPsUnder240GMs/PeakStoryDrift` | 621 | 16.4 MB | Yes |
| `GroundMotions/MirandaGroundMotions/Histories` | 240 | 20.2 MB | Yes, to compute each record's intensity |
| `GroundMotions/MirandaGroundMotions/GroundMotionInfo` | 3 | 5 KB | Yes |
| Designs, OpenSees models, other responses | not measured | | No |

**About 37 MB for slice 2.** Licence: Open Data Commons Attribution (DesignSafe
publication record). The ground-motion histories inside may carry their source's
rights, so they stay in `data/` (ignored by git) and are never redistributed.
**Waits for Arthur's OK.**

## 3. OpenSeesPy: install and speed

- Installs from PyPI on macOS arm64 (`openseespymac` 3.8.0.0, 10 MB) and Linux
  (`openseespylinux`, 86 MB) under Python 3.12.
- **Accuracy:** on broadband input at the records' 0.01 s step, Newmark's average
  acceleration put the peak 6.5% off a 40-substep reference. Substeps per sample:

  | Substeps | Worst error (elastic) | Worst error (yielding) | Time per 30 s record |
  |---|---|---|---|
  | 1 | 6.55% | 1.72% | 4.4 ms |
  | 2 | 1.94% | 0.74% | 9.5 ms |
  | 4 | 0.54% | 0.21% | 18.5 ms |
  | **5 (chosen)** | **0.34%** | **0.13%** | **24.3 ms** |
  | 10 | 0.08% | 0.03% | 47.6 ms |

- **The linear oscillator** uses an exact first-order-hold solution (scipy), within
  0.8% of a 20-substep reference, in 0.03 ms a record.
- **Expected run time on this laptop (8 cores):** a 4,000-record trial pool took 30 s,
  so the paper's 100,000 records take about 13 minutes. Active learning takes about
  143 s per start pair at full pool size, run 8 at a time, so about 6 minutes for 20.

## 4. A maintained generator for the Rezaeian and Der Kiureghian model?

None used. SimCenter's `smelt` (BSD-2) is a C++ library needing Intel MKL and IPP. We
didn't confirm that it has this exact model, and the generator is about 100 lines
that are part of what this slice reproduces. Written here, and tested against known
parameters.

## 5. The SVM regularisation constant

With 1,000 points near a noisy boundary (a stand-in for active learning's labelled
set), scikit-learn's `SVC(kernel="linear")`:

| C | Fit time | Hyperplane direction |
|---|---|---|
| 1,000 | 153 ms | (0.789, 0.542, 0.273, 0.099) |
| 10 | 6.4 ms | (0.788, 0.543, 0.272, 0.098) |
| 1 | 3.8 ms | (0.789, 0.541, 0.269, 0.109) |

C = 10, chosen before any real run.
