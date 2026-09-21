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
| C3 | **External events & macro context** (suggested by Paul's family 2026-08-05) | Only path to anticipating a turning point | **DONE 2026-08-24 — see [`EXTERNAL_EVENTS.md`](EXTERNAL_EVENTS.md).** Findings: the FY2023 −42.8% crash matches Mitsubishi Electric group **FA orders −43% YoY** (Jun-23 qtr) during the global electronics inventory correction; the FY2024 +25.1% rebound is the FA/electronics capex cycle and **not** Thai automotive (which fell 20% that year — a plausible story that is simply wrong). The indicator called both turns correctly against the model (n=2). At the Apr-2025 origin it said "up" (+28% FA orders) while the model says −8.9%. Delivered as a **documented overlay, not a fitted feature** — §6 explains why 4 years / 2 folds make it unscoreable. Provenance claims updated in README + ASSUMPTIONS §0. **DECIDED (Paul, 2026-08-24): Option B — headline stays 174,514; an external-context scenario of 211,000 units / ฿1,766.7m is published alongside as a labelled cross-check. Both inside the published range 118,093–215,741. Pipeline untouched.** |
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
171 → 242 products. Headline 176,044 → **174,507 units** (later 174,514
after §J's rounding interaction). Runtime cost: full
run ~5.5 min → ~21 min (the 49-feature fits dominate). XGBoost remains
rejected as a replacement (worse on the honest OOS test); revisit as an
additional competitor when a third fold exists.

## J. Seasonal reshape of flat-model forecasts — TESTED & ADOPTED 2026-08-24

Prompted by Paul: the FY2025 monthly profile looked flat, "like the average of
the past data". Measured: it *was* nearly flat — max/min month ratio 1.10
against 1.77 in history (March fiscal push alone is 10.9% of a year). Cause is
structural: ~1,600 products are won by models that emit a constant monthly
value by construction (moving averages, long-run mean, Croston/SBA/TSB, SES,
demand rate), and the true seasonal models are excluded until FY2025 actuals
lengthen the folds (C1).

**Fix tested through the harness** (scratchpad `reshape_test*.py`): keep every
product's annual total, redistribute its months by a seasonal index computed
from training months only. Sanity check: recomputed flat WAPE matched the
recorded per-product WAPE to 4 decimals over ~2,900 products.

| Variant (monthly WAPE, flat-model products) | Fold 1 | Fold 2 | Verdict |
|---|---|---|---|
| Flat, as delivered | 0.9840 | 0.6049 | baseline |
| Portfolio index, full strength | 0.9878 | 0.6025 | loses F1 — rejected |
| Category index, full strength | 0.9868 | 0.6083 | loses both — rejected |
| **Portfolio index, half strength** | **0.9828** | **0.6005** | **wins both — ADOPTED** |
| Category index, half strength | 0.9801 | 0.6009 | wins both; needs a category join at every call site for ~equal accuracy — passed over for robustness |

Portfolio-level monthly WAPE: Fold 2 improves 0.2259 → ~0.216; Fold 1 is
unchanged *by arithmetic* — the delivered forecast over-shoots every single
month of FY2023, so total absolute error equals the level gap regardless of
shape. (A finding in itself: when the level is wrong enough, shape is free.)

**Implementation** (commit pending): `seasonal_reshape()` in `src/forecast.py`,
applied after safeguards in the final forecast and in all three scoring paths
(`accuracy_as_delivered`, `benchmark_comparison`, `accuracy_by_level`) so
reported accuracy matches what is delivered. Only rows that are genuinely
constant are touched; floor-adjusted rows keep last year's own shape. Config:
`models.seasonal_reshape: {enabled: true, shrink: 0.5}`. Annual totals, the
headline and the range are unchanged by construction *before rounding*; the
2-decimal month rounding lands differently on reshaped values, moving the
published figures by noise only — headline 174,507 → **174,514** (+7 units,
0.004%), range 118,089–215,732 → 118,093–215,741. Deployed result (re-run
2026-08-24): monthly max/min ratio 1.10 → **1.38** (history 1.77); WAPE as
delivered 1.0216 → 1.0208 (Fold 1) and 0.6253 → **0.6220** (Fold 2);
benchmark improvement over best baseline now 17.9% / 10.1%. Intervals and
the inventory frontier are annual-total-based and unaffected.

Gains are honest but small — the real seasonal unlock remains C1 (FY2025
actuals lengthen the folds and SARIMA/Holt-Winters start competing).

## K. Inventory policy: whole units + event buffer — TESTED & ADOPTED 2026-09-21

Raised after the semi-final (Paul, 2026-09-21): the exported stock plan should be
in whole units, and the service level the business actually targets is ~90%, not
the 70% the page defaulted to. Both were tested on the withheld FY2024 year before
anything changed.

**1. The 90% target exposed a real limit.** The frontier only had a forecast
multiplier, and it topped out at 80.7% (3.0×). Extending the multiplier does not
fix that:

| Policy reaching ~90% | Average stock held |
|---|---|
| Forecast multiplier only (15×) | ~187,000 units |
| **Forecast × factor, floored by a multiple of the product's own P90 month** | **~74,000 units** |

Lumpy demand is not under-forecast, it is differently *shaped*: a product expected
at 0.3/month that occasionally takes one order of 50 needs a 150× multiplier, and a
multiplier spends that on every other product too. The event-exposure figure (A4)
already measured the right quantity — it was shown on the page but never used in
the policy. It is now. Buffers are computed from the **training window only** in
the back-test, so the test year never informs the buffer it is scored against.

Cheapest policy per service level (efficient ladder, P90-only buffer):
70% → 8.9k units · 80% → 19.7k · 85% → 29.1k · 88% → 41.0k · **90% → 74.1k** ·
92% → 135.7k · ceiling ~92–93% (about 2% of demand lands on products forecast at
zero, which no stock policy can serve).

Mixing quantiles (P75/P90/P100) was also tested and buys almost nothing (90% at
73.7k vs 74.1k), so the single published P90 is used — one rule, one sentence.

**Two honest caveats, kept visible:** (a) above ~90% the forecast stops mattering —
stock is driven by the buffer and a naive forecast needs the same 74k; the
forecast's edge is the 70–88% band (88%: 41.0k vs 49.4k naive, −17%). (b) The last
two points are expensive: 88% costs 41k, 90% costs 74k. 74k units against ~190k
annual demand is ~2.6 stock turns, a normal figure for an industrial distributor;
the old 70% point implied ~24 turns, which was unrealistically lean.

**2. Whole units, without deleting products.** Plain rounding was already
rejected (§F: it removes 1,922 products, 12.5% of volume). The stock plan now uses
a largest-remainder allocation per product (`inventory.to_whole_units`): every
month is a whole number, within one unit of the expected value, and each product's
annual total is preserved — the file adds up to the published headline exactly
(174,514; all 2,741 live products kept). The forecast deliverable itself stays as
fractional *expected* demand, for the reason documented in `config.yaml`.

**3. Inconsistencies found and fixed along the way.**
- The exported plan rounded stock **up** while the simulation behind the quoted
  service level used unrounded fractions — the downloaded plan held 9.7% more
  stock than the plan whose service level was displayed. The frontier is now
  simulated on the same whole-unit levels that are exported (`stock_targets`).
- The monthly file summed to 174,428 against a 174,514 headline (2-decimal month
  rounding); the whole-unit plan reconciles to the headline.
- Event exposure was computed by two separate copies of the same loop (pipeline
  and back-test); both now call `inventory.event_exposure`.
- The Shock Simulator also held fractional stock; it now uses the same rule.

**A row can read "forecast 0, stock held" — that is correct.** Products expected to
sell under half a unit a year round to a whole-unit forecast of 0, but still receive
a buffer. Tested: on the withheld year those products really sold (861 units, 0.45%
of demand); dropping their buffers saves 2.2% of stock and costs 0.11–0.16 points of
service — a better rate than the rest of the curve charges, so they stay. Truly
dormant products (no expected demand at all) get no stock.

Decks built before this date quote the old 70% / 1.50× / ~8,000-unit operating
point; the dashboard is the current source.

## H. Adopted after testing (2026-08-05)

| Idea | Evidence | Effect |
|---|---|---|
| **TSB candidate** (Teunter-Syntetos-Babai) in the tournament | Adopted by the tournament for 64 products, mostly Segment B; designed for demand fading to obsolescence, which Croston structurally cannot represent | Part of the both-fold WAPE improvement below |
| **Winsorise-for-fitting** (p99 of each product's non-zero months, fitting only) | Also fixed a documentation-integrity gap: config and ASSUMPTIONS §11 claimed it while the code did not do it | Combined with TSB: WAPE 1.032 → 1.027 (FY2023 fold) and 0.637 → 0.636 (FY2024 fold); headline 176,720 → 176,044 units |

## F. Tested and REJECTED — do not repeat

| Idea | Result |
|---|---|
| **E10: top-down category allocation for sparse products** (tested 2026-08-25) | Category MA6 level × product's 12-month share, scored on the sparse tier both folds: WAPE 1.155→1.394 (F1, +20.7%) and 1.171→1.462 (F2, +24.9%). Loses decisively — a sparse product's share of its category is itself too unstable to allocate with. |
| **E6: Croston alpha retune** (tested 2026-08-25, winners held fixed) | No alpha beats the current 0.10 on both folds: α=0.05 wins Fold 1 (1.115→0.980) but loses Fold 2 (0.561→0.582) — the classic one-fold trap the two-fold rule exists to catch. α=0.15 ties Fold 2, loses Fold 1. Keep 0.10. |
| **E3: Poisson/Tweedie loss for the global ML** (checked 2026-08-25) | Already in production — `global_ml` has used `loss="poisson"` since the engineered-features integration. Closed as done, not rejected. |
| **Asymmetric selection score** — punish under-forecasting harder than over (raised by Paul 2026-08-25: "the prediction should be more aggressive") | Tested offline on the scored back-test, zero model included. The dial works — under-penalty 1.5×/over 0.75× lifts FY2024 total error from −11.9% to −7.2%, extreme settings reach −0.9% — **but every setting worsens WAPE on BOTH folds** (F2 0.639→0.655→0.672; F1 1.039→1.057→1.096, with F1 over-forecast ballooning +43.6%→+54.8%). The honest cross-fold WAPE is flat (~0.797) across all settings. Conclusion: the under-forecast is not a selection bug, it is the price of accuracy on lumpy demand; buying bias costs error on every fold. **Rejected as an accuracy change.** Aggressiveness belongs in the layers built for it: the published scenario (211k on the record) and the stocking policy (stock factor > 1.0, chosen on the frontier / shock simulator). Available as a documented *policy* switch only if Mitsubishi states shortage costs more than surplus. |
| **Single-fold selection** — select models on the FY2024 fold only (raised by Paul 2026-08-25) | Measured offline from the scored back-test (2,343 products with both-fold coverage, zero-model excluded). On the year the selection never saw (FY2023): one-fold WAPE **1.472** vs two-fold **1.050** — 40% worse; the apparent FY2024 gain (0.603→0.598) is circular, the same year both chose and graded the models. 49.6% of products would flip model. FY2025 is by definition an unseen year, so the honest-OOS row is the relevant one. Also replicates ASSUMPTIONS §16's deployed result. Note the fold weights are already 1/3:2/3, so FY2024 already counts double. **Two folds stay.** |
| **E8: short-window precursor features for the global ML** (tested 2026-08-25) | Added `lines3` (order lines, last 3 months) and `lines_uptick` (activity surge vs own quarterly average) to the feature set; fitted per fold exactly as the backtest does. Loses both folds on pooled WAPE: 1.6353→1.6738 (F1, +2.35%) and 0.7259→0.7265 (F2, +0.09%); per-product win rate ~coin flip. The A3 precursor *diagnosis* stands (1.31x lift before large orders), but the signal adds nothing beyond the existing `lines12`/trend features at monthly product level. The precursor's real home remains the sales-input channel (D1), where a human reads the early trace and enters an override. |
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
