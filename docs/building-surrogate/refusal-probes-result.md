# Slice 3 refusal probes: PASS

Judged by the rule fixed in [refusal-probes.md](refusal-probes.md).

- Model: slice 2's, SHA-256 `30e616b98e6e92f9bb0f779a6a039ad785f7a6ec32cd93108943010162c77def` (checked on loading).
- Fit rows rebuilt from PRJ-2048 give the saved envelope's ranges: yes.
- Envelope distance limit: 1.81. The combination probes' distances:
  - PGA at its maximum, every Sa at its minimum: 3.54
  - PGA and short-period Sa at their minimum, long-period Sa at their maximum: 3.36
- Run time: 0.0 min on the laptop, $0.

| Kind | Probes | Expected | As expected |
|---|---|---|---|
| typical | 1 | answered | 1 of 1 |
| just inside | 24 | answered | 24 of 24 |
| just outside | 24 | refused | 24 of 24 |
| unseen combination | 2 | refused | 2 of 2 |
| absurd | 21 | refused | 21 of 21 |

## Every probe

| Kind | Probe | As expected | Answer |
|---|---|---|---|
| typical | the typical fit row | yes | answered: drift 0.0043 (90% interval 0.0026 to 0.0069) |
| just inside | storeys just inside its minimum: 1 | yes | answered: drift 0.0041 (90% interval 0.0026 to 0.0067) |
| just outside | storeys just outside its minimum: 0 | yes | refused: storeys: Input should be greater than or equal to 1 (got 0.0) |
| just inside | storeys just inside its maximum: 19 | yes | answered: drift 0.0026 (90% interval 0.0016 to 0.0043) |
| just outside | storeys just outside its maximum: 20 | yes | refused: storeys = 20 is outside the training range 1 to 19 |
| just inside | bay_width_ft just inside its minimum: 20.2 | yes | answered: drift 0.0042 (90% interval 0.0026 to 0.0068) |
| just outside | bay_width_ft just outside its minimum: 19.8 | yes | refused: bay_width_ft = 19.8 is outside the training range 20 to 40 |
| just inside | bay_width_ft just inside its maximum: 39.6 | yes | answered: drift 0.0040 (90% interval 0.0025 to 0.0065) |
| just outside | bay_width_ft just outside its maximum: 40.4 | yes | refused: bay_width_ft = 40.4 is outside the training range 20 to 40 |
| just inside | story_height_ratio just inside its minimum: 1.01 | yes | answered: drift 0.0030 (90% interval 0.0018 to 0.0048) |
| just outside | story_height_ratio just outside its minimum: 0.99 | yes | refused: story_height_ratio = 0.99 is outside the training range 1 to 2 |
| just inside | story_height_ratio just inside its maximum: 1.98 | yes | answered: drift 0.0041 (90% interval 0.0025 to 0.0067) |
| just outside | story_height_ratio just outside its maximum: 2.02 | yes | refused: story_height_ratio = 2.02 is outside the training range 1 to 2 |
| just inside | bays just inside its minimum: 1 | yes | answered: drift 0.0044 (90% interval 0.0027 to 0.0071) |
| just outside | bays just outside its minimum: 0 | yes | refused: bays: Input should be greater than or equal to 1 (got 0.0) |
| just inside | bays just inside its maximum: 5 | yes | answered: drift 0.0043 (90% interval 0.0026 to 0.0069) |
| just outside | bays just outside its maximum: 6 | yes | refused: bays = 6 is outside the training range 1 to 5 |
| just inside | floor_dead_load_psf just inside its minimum: 50.5 | yes | answered: drift 0.0029 (90% interval 0.0018 to 0.0047) |
| just outside | floor_dead_load_psf just outside its minimum: 49.5 | yes | refused: floor_dead_load_psf = 49.5 is outside the training range 50 to 110 |
| just inside | floor_dead_load_psf just inside its maximum: 108.9 | yes | answered: drift 0.0046 (90% interval 0.0029 to 0.0075) |
| just outside | floor_dead_load_psf just outside its maximum: 111.1 | yes | refused: floor_dead_load_psf = 111.1 is outside the training range 50 to 110 |
| just inside | roof_dead_load_psf just inside its minimum: 20.2 | yes | answered: drift 0.0045 (90% interval 0.0028 to 0.0073) |
| just outside | roof_dead_load_psf just outside its minimum: 19.8 | yes | refused: roof_dead_load_psf = 19.8 is outside the training range 20 to 115 |
| just inside | roof_dead_load_psf just inside its maximum: 113.8 | yes | answered: drift 0.0044 (90% interval 0.0027 to 0.0072) |
| just outside | roof_dead_load_psf just outside its maximum: 116.2 | yes | refused: roof_dead_load_psf = 116.2 is outside the training range 20 to 115 |
| just inside | pga_g just inside its minimum: 0.02969 | yes | answered: drift 0.0011 (90% interval 0.0007 to 0.0018) |
| just outside | pga_g just outside its minimum: 0.02911 | yes | refused: pga_g = 0.02911 is outside the training range 0.0294 to 0.7769 |
| just inside | pga_g just inside its maximum: 0.7692 | yes | answered: drift 0.0208 (90% interval 0.0129 to 0.0338) |
| just outside | pga_g just outside its maximum: 0.7847 | yes | refused: pga_g = 0.7847 is outside the training range 0.0294 to 0.7769 |
| just inside | sa_0_2s_g just inside its minimum: 0.06078 | yes | answered: drift 0.0011 (90% interval 0.0007 to 0.0018) |
| just outside | sa_0_2s_g just outside its minimum: 0.05957 | yes | refused: sa_0_2s_g = 0.05957 is outside the training range 0.06018 to 2.347 |
| just inside | sa_0_2s_g just inside its maximum: 2.323 | yes | answered: drift 0.0208 (90% interval 0.0129 to 0.0338) |
| just outside | sa_0_2s_g just outside its maximum: 2.37 | yes | refused: sa_0_2s_g = 2.37 is outside the training range 0.06018 to 2.347 |
| just inside | sa_0_5s_g just inside its minimum: 0.01183 | yes | answered: drift 0.0002 (90% interval 0.0001 to 0.0003) |
| just outside | sa_0_5s_g just outside its minimum: 0.0116 | yes | refused: sa_0_5s_g = 0.0116 is outside the training range 0.01171 to 1.441 |
| just inside | sa_0_5s_g just inside its maximum: 1.427 | yes | answered: drift 0.0149 (90% interval 0.0092 to 0.0241) |
| just outside | sa_0_5s_g just outside its maximum: 1.456 | yes | refused: sa_0_5s_g = 1.456 is outside the training range 0.01171 to 1.441 |
| just inside | sa_1s_g just inside its minimum: 0.008566 | yes | answered: drift 0.0002 (90% interval 0.0001 to 0.0003) |
| just outside | sa_1s_g just outside its minimum: 0.008396 | yes | refused: sa_1s_g = 0.008396 is outside the training range 0.008481 to 0.9664 |
| just inside | sa_1s_g just inside its maximum: 0.9567 | yes | answered: drift 0.0173 (90% interval 0.0106 to 0.0280) |
| just outside | sa_1s_g just outside its maximum: 0.976 | yes | refused: sa_1s_g = 0.976 is outside the training range 0.008481 to 0.9664 |
| just inside | sa_2s_g just inside its minimum: 0.0008993 | yes | answered: drift 0.0002 (90% interval 0.0001 to 0.0003) |
| just outside | sa_2s_g just outside its minimum: 0.0008815 | yes | refused: sa_2s_g = 0.0008815 is outside the training range 0.0008904 to 0.436 |
| just inside | sa_2s_g just inside its maximum: 0.4317 | yes | answered: drift 0.0222 (90% interval 0.0137 to 0.0361) |
| just outside | sa_2s_g just outside its maximum: 0.4404 | yes | refused: sa_2s_g = 0.4404 is outside the training range 0.0008904 to 0.436 |
| just inside | sa_4s_g just inside its minimum: 0.000133 | yes | answered: drift 0.0002 (90% interval 0.0001 to 0.0003) |
| just outside | sa_4s_g just outside its minimum: 0.0001304 | yes | refused: sa_4s_g = 0.0001304 is outside the training range 0.0001317 to 0.3826 |
| just inside | sa_4s_g just inside its maximum: 0.3787 | yes | answered: drift 0.0222 (90% interval 0.0137 to 0.0361) |
| just outside | sa_4s_g just outside its maximum: 0.3864 | yes | refused: sa_4s_g = 0.3864 is outside the training range 0.0001317 to 0.3826 |
| unseen combination | PGA at its maximum, every Sa at its minimum | yes | refused: this combination of inputs is unlike any training case (distance 3.54, limit 1.81), although each input is within its range |
| unseen combination | PGA and short-period Sa at their minimum, long-period Sa at their maximum | yes | refused: this combination of inputs is unlike any training case (distance 3.36, limit 1.81), although each input is within its range |
| absurd | storeys = 0 | yes | refused: storeys: Input should be greater than or equal to 1 (got 0) |
| absurd | storeys = -3 | yes | refused: storeys: Input should be greater than or equal to 1 (got -3) |
| absurd | storeys = 2.5 | yes | refused: storeys: Input should be a valid integer, got a number with a fractional part (got 2.5) |
| absurd | storeys = '١٤' | yes | refused: storeys: Value error, write it as a plain number with the digits 0-9, for example 14 or 0.35 (got '١٤') |
| absurd | storeys = '１４' | yes | refused: storeys: Value error, write it as a plain number with the digits 0-9, for example 14 or 0.35 (got '１４') |
| absurd | storeys = True | yes | refused: storeys: Value error, expected a number, not true or false (got True) |
| absurd | storeys = '1_4' | yes | refused: storeys: Value error, write it as a plain number with the digits 0-9, for example 14 or 0.35 (got '1_4') |
| absurd | storeys = '' | yes | refused: storeys: Value error, write it as a plain number with the digits 0-9, for example 14 or 0.35 (got '') |
| absurd | storeys = '14 storeys' | yes | refused: storeys: Value error, write it as a plain number with the digits 0-9, for example 14 or 0.35 (got '14 storeys') |
| absurd | storeys = 1000000000 | yes | refused: storeys = 1e+09 is outside the training range 1 to 19 |
| absurd | bay_width_ft = -30.0 | yes | refused: bay_width_ft: Input should be greater than 0 (got -30.0) |
| absurd | bay_width_ft = nan | yes | refused: bay_width_ft: Input should be a finite number (got nan) |
| absurd | bay_width_ft = 'nan' | yes | refused: bay_width_ft: Value error, write it as a plain number with the digits 0-9, for example 14 or 0.35 (got 'nan') |
| absurd | bay_width_ft = inf | yes | refused: bay_width_ft: Input should be a finite number (got inf) |
| absurd | bay_width_ft = 1e+308 | yes | refused: bay_width_ft = 1e+308 is outside the training range 20 to 40 |
| absurd | story_height_ratio = 0.0 | yes | refused: story_height_ratio: Input should be greater than 0 (got 0.0) |
| absurd | pga_g = 0.0 | yes | refused: pga_g: Input should be greater than 0 (got 0.0) |
| absurd | pga_g = -0.3 | yes | refused: pga_g: Input should be greater than 0 (got -0.3) |
| absurd | sa_1s_g = 'inf' | yes | refused: sa_1s_g: Value error, write it as a plain number with the digits 0-9, for example 14 or 0.35 (got 'inf') |
| absurd | storeys left out | yes | refused: storeys: missing; give a value |
| absurd | an unknown field, storey = 5 | yes | refused: storey: Extra inputs are not permitted (got 5) |
