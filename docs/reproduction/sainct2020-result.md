# Sainct et al. (2020), Table 2: reproduction result

Seed 2020; 20 test cases; 50 k-means groups.
Judged on the `pga` rows: **MISS**.

- Synthetic records simulated: 100000
- In the window Y <= L <= 6Y: 55451
- OpenSees didn't converge (kept, counted, not labelled): 0
- Labelled pool: 55451, of which 22.8% failed

| Curve against | Labelled records | Ours, Delta L2 % | Paper, Delta L2 % | Inside the paper's band |
|---|---|---|---|---|
| linear_peak | 100 | 4.60 +/- 1.49 | 2.8 +/- 0.9 | no |
| linear_peak | 1000 | 3.25 +/- 1.16 | 1.4 +/- 0.4 | no |
| pga | 100 | 4.02 +/- 1.46 | 2.6 +/- 1.0 | no |
| pga | 1000 | 3.00 +/- 0.38 | 1.6 +/- 0.3 | no |
| score | 100 | 7.02 +/- 3.17 | 3.8 +/- 1.6 | no |
| score | 1000 | 3.49 +/- 2.42 | 1.7 +/- 0.6 | no |
