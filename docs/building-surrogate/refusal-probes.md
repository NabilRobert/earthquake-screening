# Slice 3: refusals, with a probe suite and its pass rule

**Question:** does the screen refuse every input it shouldn't answer, with a reason a
user can act on, and answer every input it should? This is G4 and requirements
R1 to R3. ADR-009 says a refused input gets reasons, never a number.

**What is probed:** slice 2's saved model, `data/generated/building_model.pkl`. Its
SHA-256 is `30e616b98e6e92f9bb0f779a6a039ad785f7a6ec32cd93108943010162c77def`, and
its envelope is saved inside it. Nothing is retrained. The probes are built from
the model's fit rows only. The fit rows are rebuilt from PRJ-2048 with slice 2's
split (seed 2048). No test or calibration row is used.

## Fixed before any run (2026-11-04)

Before this was written, no probe had been run against the saved model, and the
envelope's limits had not been looked at. Nothing below will be tuned against the
result.

### How a screen is checked

A screen comes in as twelve named values in the units a user knows:

| Field | Meaning | Must be |
|---|---|---|
| `storeys`, `bays` | counts | whole numbers, at least 1 |
| `bay_width_ft`, `story_height_ratio`, `floor_dead_load_psf`, `roof_dead_load_psf` | the frame's design | above 0 |
| `pga_g`, `sa_0_2s_g`, `sa_0_5s_g`, `sa_1s_g`, `sa_2s_g`, `sa_4s_g` | the shaking: PGA and 5%-damped Sa, in g | above 0 |

The checks run in this order:
1. **Is each value a plain number?**
   - Text must be written with the digits 0 to 9: an optional sign, a decimal
     point and an exponent are allowed.
   - Other digits are refused, such as Arabic-Indic "١٤" or full-width "１４".
     Python and Pydantic read some of these, and a screen mustn't guess at
     what was meant.
   - Refused too: "1_4", which Pydantic reads as 14; true or false; NaN and
     infinity; empty text; a missing field; and a field the model doesn't know.
2. **Is each value possible?** Counts must be whole numbers of at least 1, and
   every other value must be above 0.
3. **Is each value within the training range?** Slice 2's envelope checks each
   input against its fit-row range, then the combination of inputs.

**One change to slice 2's envelope:** a range bound gets a tolerance of one part in
10⁹. Shaking is checked as a logarithm, so a user who types the training maximum
in g could otherwise be refused by rounding in exp and log. Every probe outside
the range is outside by 1% or more, far beyond this tolerance.

**Reasons name the field the user typed,** in the user's units. For example,
"storeys = 20 is outside the training range 1 to 19", or "sa_1s_g = 0.95 is
outside the training range 0.012 to 0.94".

### The probes

The probes are built from two kinds of fit row:
- **The typical row** is the fit row nearest the fit rows' median, in standardised
  inputs.
- **A bound row** is, for each input and each of its two bounds, the fit row with
  that input at the bound that is nearest the typical row.

Each probe changes one thing at a time:

| Kind | Probes | Built from | Expected |
|---|---|---|---|
| Typical | 1 | the typical row, as is | answered |
| Just inside | 24: each of the 12 inputs at both bounds | the bound row. Counts stay at the bound; other values move 1% inward (× 1.01 at the minimum, × 0.99 at the maximum) | answered |
| Just outside | 24: each of the 12 inputs at both bounds | the bound row. Counts go one past the bound; other values move 1% outward | refused, naming the field |
| Unseen combination | 2 | the typical row's design, with the shaking changed. (a) PGA at its training maximum with every Sa at its minimum: no real spectrum looks like that. (b) PGA, Sa at 0.2 s and Sa at 0.5 s at their minimum, Sa at 2 s and 4 s at their maximum, Sa at 1 s as in the typical row | refused as a combination unlike any training case |
| Absurd | 21 | the typical row, with one field replaced | refused, naming the field |

**The absurd probes:**
- **`storeys`:** 0, −3, 2.5, "١٤", "１４", true, "1_4", "", "14 storeys", 10⁹, and the
  field left out;
- **`bay_width_ft`:** −30, NaN, "nan", infinity, 10³⁰⁸;
- **`story_height_ratio`:** 0;
- **`pga_g`:** 0 and −0.3;
- **`sa_1s_g`:** "inf";
- **An unknown field:** `storey`, a typo, added with the value 5.

### The pass rule (G4)

**PASS** means all four of these hold. Anything else is a **miss**, and is reported
as one.

1. **The model is slice 2's.** The model file's SHA-256 matches the one above, and
   the fit rows rebuilt from PRJ-2048 give exactly the envelope's saved minimum and
   maximum for every input.
2. **Every probe expected to be refused is refused.**
   - A just-outside or absurd probe needs at least one reason that starts with
     its field's name.
   - A combination probe needs a reason that says the combination is unlike any
     training case.
3. **No refused probe gets a number.**
4. **Every probe expected to be answered is answered:** no reasons, and a finite
   drift with a finite interval around it.

**Reported, not judged:** each probe's verdict and first reason; the envelope's
distance limit; each combination probe's distance; and the run time.

**Time budget:** the run stays under 10 minutes on the laptop, at $0.

**What this doesn't test:** whether the answered probes' drifts are right. Slice 2
measured accuracy. This slice checks only who gets an answer.
