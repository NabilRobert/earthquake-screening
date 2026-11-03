# Reproducing Sainct et al. (2020): pass rule and deviations

**Paper:** Sainct, Feau, Martinez and Garnier, *Efficient seismic fragility curve
estimation by active learning on support vector machines*, Structural Safety 86
(2020); [arXiv:1810.01240](https://arxiv.org/abs/1810.01240). We use its method and
cite its numbers. No text is copied.

## The pass rule (fixed 2026-11-03, before any run on real records)

The paper's Table 2, 4-D linear SVM, mean ± standard deviation of ΔL2 over 20 start
pairs:

| Curve against | 100 labelled | 1,000 labelled |
|---|---|---|
| Score | 3.8 ± 1.6 % | 1.7 ± 0.6 % |
| **PGA** | **2.6 ± 1.0 %** | **1.6 ± 0.3 %** |
| L | 2.8 ± 0.9 % | 1.4 ± 0.4 % |

**Pass:** our mean ΔL2 for the PGA curve, over our own 20 start pairs, is inside the
paper's mean ± one standard deviation **at both 100 and 1,000**. The score and L rows
are reported next to it but not judged. Anything else is a **miss**, and it is
reported as a miss with what differed. The code holds the same rule
(`earthquake_screening/reproduction/sainct2020.py: has_passed`).

## Choices the paper doesn't fix

We fixed each of these before running on real records, and none will be tuned
against the result. Each is a named constant in the code.

| Choice | Ours | Why | Where |
|---|---|---|---|
| Number of k-means groups for ΔL2 | 50 | Not given. More groups give a larger error (each group's reference rate is noisier), so a high count avoids flattering ourselves | `FRAGILITY_GROUP_COUNT` |
| SVM regularisation C | 10 | Not given. The paper describes a maximal margin; C = 1,000 took 150 ms a fit, and C = 10 gave the same hyperplane within 0.1% in 6 ms ([spikes](../spikes.md)) | `SVM_REGULARIZATION` |
| Synthetic record length | Its parent record's length | The model's frequency drift is defined over the record's length (Eq. 2); no length is given | `SyntheticRecordSpec.duration_s` |
| Synthetic time step | 0.01 s | Not given; resolves the filter (Nyquist 50 Hz) and gives 20 samples per cycle at 5 Hz | `SYNTHETIC_TIME_STEP_S` |
| Kernel density bandwidth | scipy's Scott factor | The paper follows Kristan et al.; Scott is the standard choice with the same structure (factor × sample covariance) | `synthesis.py` |
| Logistic fit | Unpenalised | The paper fits a plain logistic regression | `failure_probabilities` |
| Damping search for the record fit | 45 values from 0.02 to 0.9, 5 simulations each | The paper matches counts "by simulation" without a grid | `DAMPING_CANDIDATES` |
| Nonlinear time integration | Newmark average acceleration, 5 substeps per sample | Within 0.4% of a 40-substep reference ([spikes](../spikes.md)) | `SUBSTEPS_PER_SAMPLE` |
| Start delay T0 of real records | Fitted with the envelope; frequencies and damping fitted on the record after it; synthetic length is the length after it | The paper fits T0 for real records and sets it to 0 when simulating, without saying how. Added after the first real record failed (pre-event noise), before any pool or result existed | `record_fit.py: fit_record` |
| Expected up-crossing rate | w_f / (2 pi), the narrow-band limit | The paper uses sigma_y' / (2 pi) with a discretisation factor of 0.75 to 1 | `fit_filter_frequencies` |
| Envelope search limits | `ENVELOPE_BOUNDS` | Without them the optimiser underflowed T2 onto T1 on a real record | `record_fit.py` |

## Known differences from the paper

- **The records.** The paper used 97 ESM records with 5.5 < M < 6.5 and R < 20 km, in
  2018, without listing them. ESM is now version 3.0. The same filter, restricted to
  events before the paper's submission (2018-09-25) and to ESM quality BEST or GOOD,
  gives **74 recordings from 18 events; both horizontal components make 148 records**
  (`data/esm_selection.csv`). The set differs from the paper's, so the fitted
  parameters, and the share of failures, will differ too.
- **8 or 9 parameters.** The paper says 9 in section 2.2 and uses 8 everywhere else
  (α1, α2, α3, T1, T2, ω0, ωn, ζf). We use the 8.
- **The engine.** The paper solved the oscillator "with a finite-difference method".
  We use OpenSees for the nonlinear run and an exact solution for the linear one.

## Result: a miss (run once, 2026-11-03)

Full table: [sainct2020-result.md](sainct2020-result.md). Seed 2020, 20 start pairs,
50 groups, judged on the PGA rows:

| Curve against PGA | Ours, Delta L2 % | Paper, Delta L2 % | Inside the band |
|---|---|---|---|
| 100 labelled | 4.02 ± 1.46 | 2.6 ± 1.0 | no |
| 1,000 labelled | 3.00 ± 0.38 | 1.6 ± 0.3 | no |

The score and L rows miss by a similar factor (1.5 to 2.3 times the paper's error).
What the method does reproduce: the error falls from 100 to 1,000 labels for every
curve, as in the paper.

**What differed in the data:**
- 134 of the 148 components fitted. The other 14 got a negative end frequency (long,
  quiet codas that a linear frequency drift can't describe) and were left out.
- 55,451 of 100,000 synthetic records fell in the window Y <= L <= 6Y, against the
  paper's 33,718, and 22.8% of them failed, against about 17%. Our records are a
  different and, by these numbers, stronger set.
- No OpenSees run failed to converge.

**Post-hoc checks** (`scripts/sainct2020_diagnostics.py`; run after seeing the result,
so they explain it and change nothing):
- **The group count isn't the reason.** Even a perfect estimator would show 1.1% with
  our 50 groups, and 0.5% with 10. We're at 3.0%.
- **Active learning isn't the reason either.** A linear SVM given 5,000 random labels
  (five times the paper's budget) reaches only 2.3 to 2.6% against PGA, and a logistic
  regression on the same four features does no better. On our records, a linear
  boundary in (L, PGA, PGV, w0) can't get near the paper's 1.6%.
- So the gap most likely comes from the inputs: a different record set, the T0 fitting
  the paper doesn't describe, and the approximate up-crossing rate. We can't tell which
  without the paper's 97 records, which it doesn't list.

**Run cost on this laptop (8 cores, $0):** downloading the records took 7 minutes,
fitting them 38, building the pool 37 and the experiment 22.
