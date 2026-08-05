# Ideas Ledger — every improvement idea, with status

Single place for every idea raised on this project. Four statuses:
**QUEUED** (to test next) · **PROVEN** (measured, awaiting adoption) ·
**FUTURE** (needs new data or approval) · **REJECTED** (tested, lost — do not repeat).

Rule of the project: nothing is adopted on plausibility. Every idea is scored on
the withheld year through the existing back-test harness first.

---

## A. Event-oriented forecasting (TESTED 2026-08-04 — results below)

The diagnosis: demand here is event-driven, not pattern-driven. 0.43% of order
lines carry 16.1% of volume; top 10 customers 68.5% (top 3: 27.6%); the two spike
months carry 31% of remaining FY2024 error; only ~10% of error is recoverable by
better model selection. These four ideas attack the event structure directly:

| # | Idea | What it does | Expected value | Effort |
|---|---|---|---|---|
| A1 | **Base + spike decomposition** | **TESTED — REJECTED.** FY2024 WAPE 0.699 vs the delivered 0.635; loses even on spike-prone products (0.724 vs 0.634, 29.8% win rate). The existing tournament candidates already cover what a flat base+expected-spike forecast offers. | — | — |
| A2 | **Customer-renewal modelling** | **TESTED — REJECTED as a forecaster; insight retained.** Churn-aware customer split beats plain product MA12 by 3.0% (WAPE 0.727 vs 0.749) — the churn signal is real — but loses clearly to the delivered pipeline (0.635), including on products >80% owned by the top-10 accounts (0.984 vs 0.937). The churn feature ("months since dominant customer's last order") moves to backlog E7 as an ML feature. | — | — |
| A3 | **Precursor test** | **TESTED — SIGNAL CONFIRMED.** 74% of large orders (>=100 units) were preceded by same-customer-product activity within 3 months, vs 57% baseline (1.31x lift; n=239 large lines). Not strong enough to time spikes statistically, but it validates the sales-input channel (D1): the information exists before the order lands. Candidate ML feature (E8). | Knowledge | Done |
| A4 | **Event-exposure buffers** | For spike timing that cannot be predicted: per-product spike-size distribution (e.g. P90 monthly order) surfaced as an "event risk" column on the Inventory Planning page and in the recommended-stock CSV. Honest — plans for events without pretending to time them. | Medium (business value) | Low |

## B. Proven, awaiting adoption

| # | Idea | Evidence | Status |
|---|---|---|---|
| B1 | **Ensemble top-2** | Isolated test: WAPE 0.816 → 0.780 (−4.3%). **Deployed test (2026-08-04): a wash** — Fold 2 WAPE unchanged at 0.637, bias slightly better (−19.8% → −18.3%), Fold 1 clearly worse (+46.7% → +49.8%). The strong two-fold selection had already absorbed the gain blending offers weaker selections. | **NOT adopted.** Fully implemented behind `models.ensemble_top_k` (one-line switch); revisit whenever selection rests on a single fold, e.g. mid-year refreshes |

## C. Needs new data or approval (FUTURE)

| # | Idea | Why it matters | Blocker |
|---|---|---|---|
| C1 | **FY2025 actuals** | +25% history; seasonal models (SARIMA ranked #1 of ~24 on dense products in its one usable fold) unlock automatically via `folds: auto`; first true validation against the published range 118,637–216,138 | Time — arrives after grading |
| C2 | **Sales pipeline / quotation data** | The Aug/Oct spikes were known to sales before they landed. Top-50 error products carry 39.4% of error. Largest single lever on the whole project. | Mitsubishi must provide |
| C3 | **One external macro series** (Thai PMI / industrial production / auto production) | Only path to anticipating a turning point like the −43% FY2023 crash | Approval to use external data (project currently uses supplied files only, deliberately) |
| C4 | **Stock-availability history** | Sales are censored by stock: in shortage periods, lost sales were never recorded, so history understates true demand exactly where it matters. With availability data, censored months can be corrected. | Mitsubishi must provide |
| C5 | **Time-series foundation models** (Chronos/TimesFM/Moirai, zero-shot) | Potentially strong on short series; a differentiator | Heavy dependency; hard to validate with only 2 folds; revisit later |

## D. Process improvements (no modelling)

| # | Idea | Value |
|---|---|---|
| D1 | **Monthly sales-override channel**: 30-min review flagging known large orders for the top ~50 error products; entered as manual overrides | Larger than all modelling items combined |
| D2 | **Rolling monthly re-forecasts** (upload feature already supports; ~2 min per run) | Short-horizon accuracy always beats 12-months-out |
| D3 | Present segments as **ABC classification** (standard inventory vocabulary) | Recognition/credibility with Mitsubishi |

## E. Smaller modelling refinements (backlog)

| # | Idea | Note |
|---|---|---|
| E1 | Quantile global ML (pinball loss p10/p50/p90) | Also yields per-product intervals from the model itself; `quantile` param already exists in `GlobalMLForecaster` |
| E2 | Train global ML on segments A+B only (or segment feature) | `sample_mask` parameter already exists, currently unused |
| E3 | Tweedie loss option for the global model | For zero-inflated continuous-ish demand |
| E4 | Hyperparameter search routed through the back-test harness | Guarded by the same selection-margin rule |
| E5 | Rolling-origin fold scheme (all training windows ≥30 months) | Becomes possible at 60 months of history; fairer than annual folds |
| E6 | Croston alpha tuning | Config knob already exposed (`models.croston_alpha`) |
| E7 | Churn feature for the global ML model: "months since the dominant customer's last order" | The churn signal is real (+3.0% vs plain MA12 in the A2 test); the customer axis carries information the product axis lacks |
| E8 | Precursor feature: same-pair order activity in the prior 1–3 months | 1.31x lift measured in the A3 test |

## F. Tested and REJECTED — do not repeat

| Idea | Result |
|---|---|
| Hierarchical reconciliation to category totals | Combined score 1.16 → 1.46 (worse). Off; evidence in `reconciliation_decision.json`. |
| Distributional (quantile) stocking policy | Dominated by the point-forecast policy at every inventory level tested. |
| Seasonal models competing on one fold only | FY2024 total error −19.2% → −21.7% (worse); skipping the hard fold flatters the average. Reverted. |
| `naive` / `moving_average_3` for sparse segments | Structural zeros — forecast nothing for products that sell. Excluded by segment rules. |
| Per-month integer rounding | Deleted 1,922 real products (12.5% of volume). Reverted to fractional months. |
| Median line price for revenue | Overstated revenue ~20%; replaced by quantity-weighted price (validated +2.3% on FY2024). |
| Summing per-product interval bounds | Gave a meaningless 74k–367k band; portfolio range now from total-level evidence. |
| Deep learning (LSTM/DeepAR/N-BEATS) | Not attempted, deliberately: 48 monthly points, median 3 non-zero. Reconsider at C1/C5. |

## G. Open business questions (change numbers via one config line)

| Question | Exposure | Switch |
|---|---|---|
| Is Base Price invoiced or pre-VIP-discount? | **฿461m / 22.5% of revenue** | `pricing.vip_discount` |
| What does order status `Closed` mean? | 14.5% of volume | `cleaning.revenue_statuses` |
| FY label convention (FY2025 vs FY2026) | Label only | `fiscal.forecast_fy` |
