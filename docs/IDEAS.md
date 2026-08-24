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
| A4 | **Event-exposure buffers** | **IMPLEMENTED 2026-08-05.** Per-product P90 single-month demand computed from each product's own non-zero months; shown on the Inventory Planning page (top-exposure table) and included as a column in the recommended-stock download. Plans for events without pretending to time them — the constructive response to the A1–A3 findings. | Done | Done |

## B. Proven, awaiting adoption

| # | Idea | Evidence | Status |
|---|---|---|---|
| B1 | **Ensemble top-2** | Isolated test: WAPE 0.816 → 0.780 (−4.3%). **Deployed test (2026-08-04): a wash** — Fold 2 WAPE unchanged at 0.637, bias slightly better (−19.8% → −18.3%), Fold 1 clearly worse (+46.7% → +49.8%). The strong two-fold selection had already absorbed the gain blending offers weaker selections. | **NOT adopted.** Fully implemented behind `models.ensemble_top_k` (one-line switch); revisit whenever selection rests on a single fold, e.g. mid-year refreshes |

## C. Needs new data or approval (FUTURE)

| # | Idea | Why it matters | Blocker |
|---|---|---|---|
| C1 | **FY2025 actuals** | +25% history; seasonal models (SARIMA ranked #1 of ~24 on dense products in its one usable fold) unlock automatically via `folds: auto`; first true validation against the published range 118,637–216,138 | Time — arrives after grading |
| C2 | **Sales pipeline / quotation data** | The Aug/Oct spikes were known to sales before they landed. Top-50 error products carry 39.4% of error. Largest single lever on the whole project. | Mitsubishi must provide |
| C3 | **External events & macro context** (suggested by Paul's family 2026-08-05: web-search real occurrences — Thai PMI, auto/electronics cycle, plant news — to explain FY2023/FY2024 turning points and inform FY2025) | Only path to anticipating a turning point; to be applied first as a documented overlay, separate from the supplied data | Queued for before the semi-final. Requires updating the "no external data" provenance claim in README / ASSUMPTIONS §0 / dashboard wherever it appears |
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
| E9 | Order-count and customer-count monthly features for the global ML model | The order-line data carries frequency/breadth signals the quantity sum discards |
| E10 | Top-down category-share model for sparse products (category forecast x recent product share) | Different from the rejected reconciliation: allocation, not scaling |

## I. XGBoost + feature engineering (professor's advice) — TESTED 2026-08-16, isolated

Prompted by the course professor: (1) XGBoost is the popular choice for
forecasting; (2) feature engineering matters — derive new columns from the
data. Tested in the scratchpad (`ml_experiment.py`) without touching the
pipeline. Note the pipeline already had a gradient-boosting model of the same
family (`global_ml`, sklearn HistGradientBoosting); the question was whether a
different library or richer features would improve it.

**19 engineered features tested** (all computed from data available at the
origin): target-month same-month-last-year / two-years-ago, own and category
seasonal index at the target month, horizon; spike ratio, spike count,
coefficient of variation, trend ratios; **customer side** — distinct customers
last 12m, top-customer share, VIP quantity share, order lines and quantity per
line (E7/E9 backlog items); production status and zero-list-price flags.

| Variant | Standalone WAPE F1 / F2 | Tournament delivered F1 / F2 | Honest OOS (select F1 → measure F2) |
|---|---|---|---|
| current global_ml (HGB, 30 feats) | 1.877 / 0.755 | 1.045 / 0.683 | 0.810 |
| **HGB + engineered feats** | 1.635 / 0.726 | **1.031 / 0.673** | **0.804** |
| XGBoost, current feats | 1.753 / 0.745 | 1.032 / 0.682 | 0.805 |
| XGBoost + engineered feats | **1.486 / 0.719** | 1.025 / 0.681 | 0.813 (worse) |
| LightGBM + engineered feats | 1.505 / 0.736 | 1.028 / 0.682 | 0.806 |

(Tournament numbers here come from the experiment's own re-scoring function,
so they are comparable across rows but not identical to the pipeline's
as-delivered figures.)

**Findings.** (a) Feature engineering is the lever, not the library: the
engineered features improve every algorithm by 8–20% standalone; swapping HGB
for XGBoost on the same features gains ~2–7%. (b) As a standalone model
XGBoost+features is best — but in the tournament it is the only variant that
gets *worse* on the honest out-of-sample test (0.813 vs 0.810), i.e. its extra
wins on the selection folds do not carry forward. (c) **HGB + engineered
features is the only variant that improves on all three tests** (−1.4% F1,
−1.5% F2, −0.7% OOS). Gains are modest because the ML model wins only 3–5% of
products; the ~10% selection-headroom ceiling still applies.

**Status: ADOPTED — integrated 2026-08-24 (Paul's go-ahead after the
presentation).** `AuxPanels` in `src/global_ml.py` builds the customer/order-
line panels once per run; both back-test and final forecast use them. Deployed
result (as delivered, incl. safeguards): WAPE 1.027 → **1.022** (FY2023 fold)
and 0.636 → **0.625** (FY2024 fold); FY2024 bias −20.0% → −20.6%; ML wins
171 → 242 products. Headline 176,044 → **174,507 units**. Runtime cost: full
run ~5.5 min → ~21 min (the 49-feature fits dominate). XGBoost remains
rejected as a replacement (worse on the honest OOS test); revisit as an
additional competitor when a third fold exists.

## H. Adopted after testing (2026-08-05)

| Idea | Evidence | Effect |
|---|---|---|
| **TSB candidate** (Teunter-Syntetos-Babai) in the tournament | Adopted by the tournament for 64 products, mostly Segment B; designed for demand fading to obsolescence, which Croston structurally cannot represent | Part of the both-fold WAPE improvement below |
| **Winsorise-for-fitting** (p99 of each product's non-zero months, fitting only) | Also fixed a documentation-integrity gap: config and ASSUMPTIONS §11 claimed it while the code did not do it | Combined with TSB: WAPE 1.032 → 1.027 (FY2023 fold) and 0.637 → 0.636 (FY2024 fold); headline 176,720 → 176,044 units |

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

| Question | Status |
|---|---|
| Is Base Price pre-VIP-discount? (**~฿440m**) | DECIDED 2026-08-16 — discount APPLIED (`pricing.vip_discount: 0.37`): dictionary states it, paired test shows it is not embedded in Base Price. Undiscounted figure still reported alongside; flip to 0.0 if Mitsubishi says Base Price is the invoiced amount |
| Order status `Closed` | RESOLVED — "won't be fulfilled or billed" (authoritative sheet). Demand: all statuses; revenue: excludes Closed |
| Grading definition of FY2025 actuals | CLOSED per Paul — not asked; aligned to the supplied extract's structure |
| FY label convention | Label only — `fiscal.forecast_fy` |
