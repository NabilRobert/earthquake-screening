# Phase 1: Problem Brief

**Project:** instant earthquake screening for buildings. A fast stand-in
model (a *surrogate*) is trained on nonlinear earthquake simulations. Given
a building's basic description, it estimates in seconds how badly the
building is likely to be damaged, where a full analysis takes a specialist
and long computer runs. An agent layer lets someone describe the building
in plain language and get a short report. It is the eleventh portfolio
piece.

The working name is the repo's, `earthquake-screening`.

**Settled with Arthur (2026-11-03):** it is built as an **open portfolio and
research piece first**. Selling it is a later decision that waits on the
licence questions in [Phase 5](phase-5-architecture-decisions.md) (ADR-001).

## Problem

**For the people who'd use it:**
- **A full seismic check is slow and needs a specialist.** Nonlinear
  time-history analysis means building a model, choosing ground motions and
  running many simulations. One building takes an engineer days to weeks.
- **Portfolios are large.** An insurer or a bank with thousands of
  buildings in Jakarta, Istanbul or Tokyo can't afford that for each one.
  Today they use coarse tables by building class (Hazus-style), which
  ignore the building's own height, period or strength.
- **A fast answer that is quietly wrong is dangerous.** A screen that says
  "low risk" outside what it was trained on is worse than no screen.

*To verify, not assumed:* that the surrogate is meaningfully more accurate
than the class tables on held-out buildings. If it isn't, the pitch fails
(goal G3).

**For the portfolio:**
- The other projects are agent, backend and tooling work. None needs
  specialist knowledge from another field, or slow, maths-heavy work.
- A reviewer can't yet see Arthur train a model and report its error
  honestly: on held-out data, with uncertainty, and with refusals when an
  input is out of range.
- Nothing yet reproduces a published result. This one starts by doing that.

## Users

| Party | Who | Wants |
|---|---|---|
| **Screening user (later)** | An analyst at an insurer, a bank with property loans, or a developer, in Indonesia, Japan, Turkey or California | A ranked list of which buildings need a real engineer first |
| **Structural engineer** | Signs off any real decision; reviews the method | A transparent method, stated limits, and a model it can check |
| **Reviewer / hiring team** | Japan-style employers | A reproduction that matches the paper, honest accuracy, refusals, tests, documentation |
| **Arthur (builder)** | Learns the domain from papers | A project he can explain end to end |

## Positioning: what makes this one different

1. **Reproduce first.** Slice 1 reproduces one published surrogate or
   fragility result and reports how close it gets. Nothing new is claimed
   until that matches.
2. **Accuracy is measured against full simulation, on held-out buildings.**
   Not on random rows from buildings the model has seen.
3. **It refuses outside its training range.** An input outside the
   envelope of the training buildings gets "out of range: ask an engineer",
   not a number.
4. **Screening only.** Every report says it is not a structural assessment,
   and that an engineer signs off any decision.
5. **The licences were read.** OpenSees, OpenQuake, ETABS and every dataset,
   from their own texts, before any code (Phase 5).

## Goals and how they're measured

| # | Goal | Target | Measured by |
|---|---|---|---|
| G1 | The reproduction matches the paper | Within a tolerance set before running (slice 1) | Reproduction report, scripted |
| G2 | Honest accuracy | Error on held-out *buildings*, with intervals that cover at their stated rate (for example 90% intervals cover 88–92%) | Held-out test in CI on a small fixed set |
| G3 | Better than the coarse baseline | Lower error than the Hazus-style class table on the same held-out buildings | Same test |
| G4 | Refuses outside the range | 100% of out-of-envelope probes refused; 0 refusals inside it | Probe suite |
| G5 | Fast | One screening in under 1 s on a laptop CPU | Timed test |
| G6 | $0 compute | Training data generated on a laptop or the small shared server, within a time budget set in slice 2 | Run logs |
| G7 | The agent doesn't invent numbers | Every number in a report comes from the surrogate's output | Inspect AI eval with no paid model (slice 5) |

## Non-goals (v1)

- **Selling it, or any real decision.** Portfolio and research only
  (ADR-001).
- **Running ETABS, or an ETABS add-on.** Recorded as the commercial path
  (ADR-004), not built.
- **A site-specific hazard calculation.** v1 takes the shaking intensity as
  an input, or uses published hazard values with a clear licence.
- **Every building type.** v1 covers one structural system (chosen with the
  paper in slice 1).
- **Loss in money.** v1 stops at damage (drift and damage-state
  probabilities). Repair cost (FEMA P-58 style) is later at the earliest.
- **Public sign-up or a hosted service.**

## Open questions

See the end of [Phase 5](phase-5-architecture-decisions.md#open-questions).
