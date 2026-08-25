# Improvement Roadmap — measured, not speculative

This project will improve over time. This document states **where the improvement
will come from and how much is available**, based on measurements against the
withheld FY2024 — not on hope. Suitable for presenting as the "next steps" section
of a progress report.

---

## The honest ceiling first

**How much can smarter modelling alone still recover?** We measured it: if, with
perfect hindsight, the best of the ~24 candidate models had been chosen for every
single product, FY2024 WAPE would improve from 0.630 to 0.568 — meaning **at most
10% of the remaining error is recoverable by any model-selection scheme, however
clever.** The other 90% is demand behaviour that none of the candidate methods
captured, because the information explaining it is not in the sales history.

This is why the roadmap below is mostly *not* about new algorithms.

---

## Roadmap, ordered by expected impact

### 1. New data arrives free: FY2025 actuals (largest modelling gain, zero effort)

When the FY2025 actual figures become available, adding them to `data/raw/` and
re-running does three things at once:

- **+25% more training history** (48 → 60 months).
- **The seasonal models unlock automatically.** Holt-Winters and SARIMA need 25–28
  training months per fold; with `backtest.folds: auto` every fold then clears that
  bar. SARIMA already ranked best of ~24 methods on the dense products in the one
  fold it could run — it is currently excluded only because the earliest fold is
  too short. No code change required; this is already built and verified.
- **First true validation**: the FY2025 forecast (174,514 units; range
  118,093–215,741) gets compared against reality. Whether the outturn lands inside
  the published range is the single most informative test this project can receive.

### 2. Target the error where it concentrates (largest total gain, no modelling)

The remaining FY2024 error is extraordinarily concentrated:

| Slice | Share of ALL remaining error |
|---|---|
| Top 10 products | **22.5%** |
| Top 50 products | 39.4% |
| Top 200 products | 62.5% |
| The two spike months (Aug + Oct 2024) | **31%** |
| PLC category alone | 44% |

The two worst months were driven by a handful of very large orders — exactly the
kind of demand that **no statistical method predicts but the sales pipeline knows
about in advance**. The highest-value improvement to this system is therefore an
*information* channel, not an algorithm: a monthly 30-minute review where sales
flags known large orders and contract changes for the top ~50 error-carrying
products, entered as manual overrides. Realistic impact: larger than every
modelling item below combined.

### 3. Ensemble forecasting (tested — it works)

Averaging the top-2 models per product instead of using the single winner was
tested honestly (models chosen on the FY2023 fold only, scored on FY2024 only):

| Approach | FY2024 WAPE |
|---|---|
| Single best model per product | 0.816 |
| **Average of top-2 models** | **0.780 (−4.3% error)** |

A modest, real, essentially free gain — a classic result in forecasting (forecast
combinations beat single models). Planned as the first modelling change after the
progress review, so that reported numbers stay stable until then.

### 4. External demand drivers — DONE as an overlay (2026-08-24)

Nothing in the supplied files explains the −43% FY2023 crash or the +25% FY2024
rebound — and these turning points dominate the error. Approved and researched;
full write-up in [`docs/EXTERNAL_EVENTS.md`](EXTERNAL_EVENTS.md).

**What it found.** FY2023 matches the global electronics inventory correction
almost exactly — Mitsubishi Electric group Factory Automation orders fell **43%
YoY** in the Jun-2023 quarter against this portfolio's **−42.8%** year. FY2024's
rebound is the FA/electronics capital-expenditure recovery, and specifically **not**
Thai automotive, which fell 20% that same year — the obvious story is the wrong one.
Read at each forecast origin, group FA order growth called both turning points
correctly, in both cases against the model.

**Why it is an overlay and not a feature.** With four fiscal years and two folds, an
annual macro regressor cannot be scored on a withheld year, so adopting it as a
fitted input would violate this project's rule that nothing is adopted on
plausibility. The monthly Thai series (MPI, PMI) are long enough to fit but measure
the wrong axis — they moved opposite to this portfolio in FY2024.

**What it changes.** No published number. It supplies the missing explanation for
the dominant error source, and it flags that the FY2025 point forecast sits low in
its own range with four independent routes pointing above it.

### 5. Process change: rolling monthly re-forecasts

The current setup forecasts 12 months once a year. Re-running monthly as orders
arrive (the upload feature already supports this in ~2 minutes) means month 8's
forecast uses 7 more months of information than it does today. Short-horizon
accuracy is always materially better than 12-months-out accuracy.

---

## What was tried and did NOT work (so nobody repeats it)

| Attempt | Result |
|---|---|
| Hierarchical reconciliation to category totals | Made accuracy worse (combined score 1.16 → 1.46). Off, decision recorded in `reconciliation_decision.json`. |
| Distributional (quantile-based) stocking policy | Beaten by the point-forecast policy at every inventory level tested. |
| Allowing seasonal models to compete on one fold only | Degraded FY2024 total error −19.2% → −21.7%; reverted. |
| Deep learning | Not attempted, deliberately: 48 monthly points with a median of 3 non-zero would overfit. Reconsider only after several more years of data. |

---

## Suggested wording for the progress report

> "The system currently beats naive planning by 7–15% on error and by 4–6
> percentage points of service level at equal stock. We have measured the
> remaining headroom: only ~10% of current error is recoverable by better
> algorithms, so our roadmap prioritises the levers the measurement points to —
> the FY2025 actuals that unlock seasonal models automatically, a sales-input
> channel for the ~50 products that carry 40% of the error, and a tested ensemble
> method worth a further 4%. Each item is already scoped and none requires
> rebuilding the system."
