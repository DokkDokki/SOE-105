# Assumptions & Decisions Log

Every judgement call made in this project, with the evidence behind it. Written so
the team can defend the numbers to Mitsubishi, and so any assumption can be
challenged and changed in `config.yaml` without touching code.

---

## 0. Data provenance — nothing outside the supplied files

**Every number in this project derives solely from the three files Mitsubishi
supplied:** `Sale Order History.xlsx`, `Item.xlsx` and `Customer.xlsx`.

- No internet access, no API calls, no downloads. The codebase imports **no**
  networking library — verifiable by searching the source for `requests`,
  `urllib`, `http`, or any URL.
- No market indices, economic indicators, competitor data, benchmark datasets or
  third-party sources.
- No figures typed in from outside, and **no FY2025 actuals** — those are held out
  for grading and are never inferred, estimated from outside sources, or fabricated.

**On the word "global".** The machine-learning component is described in forecasting
literature as a *global* model. This is standard terminology meaning **one model
fitted across all the series in the dataset**, as opposed to a *local* model fitted
separately to each series. It refers to the modelling structure, **not** to
global/world data. "Trained on all products" means all products in this dataset. The
dashboard avoids the jargon entirely and calls it "Machine learning (learns from all
products in this dataset together)".

*(Note: earlier discussion suggested that adding an external series such as a
manufacturing index could improve future accuracy. That was a suggestion for
possible future work only — it is **not** implemented and no such data exists
anywhere in this project.)*

---

## 1. The fiscal year is April → March

**Decision.** FY2021 = Apr 2021 – Mar 2022 … FY2024 = Apr 2024 – Mar 2025.
Forecast target **FY2025 = Apr 2025 – Mar 2026**.

**Evidence — detected from the data, not assumed.**

| Signal | Observation |
|---|---|
| Start boundary | Extract begins exactly **2021-04-01** |
| End boundary | Extract ends exactly **2025-03-31** |
| Coverage | Exactly **48 months = 4 complete years**, no gaps, no missing months |
| Seasonal peak | **March = 10.9%** of annual volume vs 8.3% if flat — the fiscal year-end push |

A calendar-year extract would not land on those boundaries. Detection confidence:
**high**.

**Risk.** Some organisations would label Apr 2025 – Mar 2026 as "FY2026". If the
grader uses that convention, only the *label* changes, not the numbers. Controlled
by `fiscal.forecast_fy`.

---

## 2. Order status: demand counts everything, revenue counts only invoiced

**The definition — resolved 2026-08-05.** The dictionary carries two readings of
`Closed`, but they describe different artifacts: the *Sales Sheet* tab ("completed
and officially closed") documents a different form, while the *Sales Order History*
tab documents the very file this project uses and is therefore authoritative:
**"The sales order is closed and won't be fulfilled or billed."** `Closed` is
cancelled work — and it is **14.5% of FY2024 quantity**, so its treatment matters.

**Decision.**

| Purpose | Statuses counted | Rationale |
|---|---|---|
| **Demand** | All statuses | A cancelled order still expressed real customer demand — the signal inventory planning needs. It also aligns the forecast target with the structure of the supplied extract: the graded FY2025 actuals will most plausibly be the same kind of file, in which every status appears. |
| **Revenue** | `Billed`, `Pending Billing` | With `Closed` now confirmed as never-billed, excluding it from revenue is definitively correct, not merely cautious. |

The bridge between the two is a per-product **billed share** (catalogue average
0.854), applied to the quantity forecast before pricing.

Both series are published, so nothing is hidden by the choice.
Controlled by `cleaning.demand_statuses` / `cleaning.revenue_statuses`.

---

## 3. Exact duplicate order lines were KEPT

**The finding.** 6,865 exactly duplicated rows, carrying **1.80% of total quantity**.

**Why this is not an export artifact — four forensic tests:**

| Test | Result | Interpretation |
|---|---|---|
| Whole-order vs partial duplication | **2,568 of 2,674** affected orders duplicate only *some* lines | An export/copy bug would clone the **entire order block** uniformly. It does not. |
| Distribution across time | Evenly spread: FY21 4.4%, FY22 5.2%, FY23 4.0%, FY24 3.8% | A one-off bad export would cluster in a single period or batch. It does not. |
| Quantity pattern | **65% carry qty = 1** (vs 55% overall) | The classic signature of an ERP entering N units as N separate lines (per serial number or delivery schedule line). |
| SO-number collisions | Only **4** duplicated rows sit in orders sharing an SO number | Not an ID-collision artifact. |

**Decision: KEEP.** Dropping them would understate real demand by 1.8%.

**Reversible:** set `cleaning.drop_exact_duplicates: true`. The pipeline reports
the impact either way.

**Robustness test (2026-08-06).** A team member independently prepared cleaned
source files whose only substantive change was removing these 6,865 duplicate
lines (plus cosmetic header trims). We ran the full pipeline on both versions:

| | Original (kept) | Duplicates removed |
|---|---|---|
| FY2025 forecast, units | 175,969 | 172,916 (−1.7%) |
| FY2025 forecast, revenue | ฿1,907.3m | ฿1,880.3m (−1.4%) |
| WAPE, FY2024 fold (as delivered) | 0.636 | 0.624 |
| Bias, FY2024 fold | −20.0% | −19.8% |

The forecast is robust to this cleaning choice (≤2% movement everywhere).
We kept the originals because (a) the forensic evidence above says the lines
are genuine demand, and (b) the grading actuals come from the source ERP,
which would include them. The cleaned files' marginally lower WAPE is measured
against their own reduced actuals, so it is not evidence the removal is more
correct.

---

## 4. Price = 0 is never treated as ฿0 revenue

**Business meaning.** A zero `Base Price` means the price is requested from the
Japan branch for that order and varies. It is **not** a free product.

**Scale of the issue.**

| | Count |
|---|---|
| Order lines priced at 0 | 4,830 (1.78%) |
| Distinct products touched | 1,081 |
| Products **always** priced 0 | 79 (only **0.05% of total quantity**) |
| Of those, having a master list price | 74 |
| **Genuinely unrecoverable** | **5 products** |

**Resolution cascade** (first rule that succeeds wins):

1. `realised` — **quantity-weighted** average price achieved in the last 12 months
2. `realised_all` — same, over the product's whole history
3. `list_adjusted` — master list price × catalogue realised/list ratio (**0.885**, computed from this dataset)
4. `PRICE_TBD` — revenue left **blank**, never guessed. Quantity still forecast.

**Why quantity-weighted and not median.** Using the median of *line* prices
over-stated forecast revenue by ~20%, because bulk lines sold at lower prices carry
most of the volume but count only once each in a median. Revenue is earned per
unit, so the correct price is total value ÷ total units.

**Flagging.** Any product whose history contains *any* zero price is marked
`PRICE_VARIABLE_ESTIMATED` even when a price was recovered, so an estimate is never
mistaken for a firm price. The 5 unrecoverable products are marked `PRICE_TBD` and
shown separately in the dashboard.

**Validation.** Running FY2024's *actual* quantities through exactly these rules
reproduces FY2024's actual invoiced revenue to within **+2.3%**.

---

## 5. Product segmentation drives model choice

**The problem.** The **median product sells in only 3 of 48 months**. Holt-Winters
and SARIMA require ≥2 full seasonal cycles. Applying them catalogue-wide would
produce confident nonsense for ~70% of products.

| Segment | Definition | Products | Share of FY24 volume | Methods allowed |
|---|---|---|---|---|
| **A** Regular seller | ≥24 active months | 932 | 82% | All: baselines, Holt-Winters ×4, SARIMA ×3, pooled ML |
| **B** Occasional | 6–23 active months | 1,032 | 11% | Moving averages, drift, SES/Holt, Croston/SBA, category seasonal profile |
| **C** Rare / lumpy | 1–5 active months | 1,257 | 8% | Croston, Croston/SBA, demand-rate only |
| **D** No recent sales | none in FY2024 | 2,485 | 0% | Forecast zero, labelled |

**`naive` ("repeat last month") is excluded below Segment A.** On a product that
sells in only some months, last month is very often zero — naive quietly forecast a
whole year of nothing for products that genuinely sell.

**`moving_average_3` is excluded from Segment C.** A 3-month average of a product
that sells twice a year is almost always zero.

---

## 6. The selection metric penalises bias, not just error

**This is the most important modelling decision in the project.**

**The problem.** Minimising absolute error (MAE/WAPE) on intermittent demand drives
the forecast towards the **median** of the series. For a product selling in 4 months
out of 12, the median is **zero**. Selecting purely on WAPE produced:

- a **−30% systematic under-forecast** of total units on Fold 2, and
- **1,922 products** that genuinely sold in FY2024 forecast at zero.

For a company already suffering **stock shortages**, that is precisely the wrong
direction of error.

**The fix.**

```
score = WAPE + bias_weight × |forecast_total − actual_total| / actual_total
```

with `bias_weight = 1.0`. A zero forecast for a product that does sell now scores
1.0 + 1.0 = 2.0 and can no longer win by default.

**Effect:** systematic bias improved from **−30% → −19%**; products with a non-zero
forecast rose from 1,299 → 3,217.

---

## 7. Monthly quantities are kept fractional

**Decision.** Monthly forecasts carry 2 decimal places; only annual totals are
rounded to whole units.

**Why.** Rounding each month to an integer deleted **1,922 real products and 12.5%
of total volume** — a product genuinely expected to sell 0.4 units a month rounds to
0 in every single month and vanishes from the plan. A monthly figure of 0.4 is
meaningful information ("about 5 a year"); a zero is not.

Controlled by `output.round_quantities`.

---

## 8. Two business safeguards

### Shortage floor (`safeguards.never_zero_if_sold_recently`)

No product that demonstrably sold in the last 12 months is forecast below
**25% of its own last-year volume** (`safeguards.min_fraction_of_last_year`).

**Why the back-test cannot catch this on its own:** for a lumpy product whose test
year happened to contain no sales, a near-zero forecast scores *well*. Near-zero wins
on the evidence — but planning near-zero for a product that does sell guarantees a
stockout.

**Revised after code review.** The original floor triggered only on an exactly-zero
annual forecast, which let the worst cases through: a product that sold 1,786 units
was forecast at 12, and one that sold 51 units across 10 months was forecast at
0.09 — both passed a "greater than zero" test. The original design also floored
zero-forecast products at 100% of last year, creating an indefensible discontinuity
(a forecast of 0.00 received the full floor; a forecast of 0.01 received nothing).
The proportional floor removes both defects, and back-test WAPE improved on both
folds (1.051 to 1.032, and 0.642 to 0.637). The reference is last year's own volume,
deliberately **not** a lifetime average, which turned a product selling 1 unit/year
into a forecast of 24. Applied to **1,276 products**.

### Surplus cap (`safeguards.max_growth_vs_recent_peak`)

No product is forecast above **1.5×** the largest annual volume it achieved in the
last three fiscal years.

**Why.** "Repeat last month" and straight-line trend can take one unusually large
final month and project it across twelve — one product jumped from 290 units to 948
that way, which quietly becomes surplus stock. There is no evidence for volumes
beyond historical peaks. Applied to **119 products**.

Both are reported in the dashboard and in `outputs/`, and both can be disabled.

---

## 9. Back-testing uses two folds, not one

| Fold | Trained on | Forecasts | Weight |
|---|---|---|---|
| Fold 1 | FY2021–FY2022 | FY2023 | 0.35 |
| Fold 2 | FY2021–FY2023 | FY2024 | 0.65 |

**Why Fold 1 was added.** The brief specified Fold 2 only. But Fold 2's test year is
the **+25% rebound**, so on its own it rewards whichever model happened to guess
"up" — for the wrong reason. Fold 1's test year is the **−43% collapse**. Requiring
a model to perform across one crash and one rebound is a far harder, more honest
test. Fold 2 still carries the greater weight because it is more recent.

**Anti-overfitting guards:**

- **Simplicity margin (5%)** — a complex model only wins if it beats the best simple
  baseline by a clear margin. A 12-point test window can crown a complex model by luck.
- **No leakage** — segment assignment during back-testing uses only the training
  window, never full history.
- **Fold-weight correction** — weights only count folds where a score actually
  exists, so a model that *failed* on one fold cannot look better than one that
  worked on both.

---

## 10. Hierarchical reconciliation: tested, then rejected

Product forecasts can be scaled to sum to independently-forecast Product Category 1
totals — standard practice for sparse catalogues, since category series are dense
even when their members are not.

**It was implemented, back-tested, and it made accuracy worse:**

| | Bottom-up | Reconciled |
|---|---|---|
| Combined score | **1.161** | 1.464 |
| WAPE | **0.848** | 1.029 |

**Decision: switched off.** The verdict and its evidence are written to
`outputs/reconciliation_decision.json` on every run, so this is an auditable
measurement rather than a matter of taste. If future data changes the verdict, the
pipeline switches it on automatically.

---

## 11. Outliers are not deleted

Quantity is heavily right-skewed (median 1, p99 = 30, max 6,000). Large orders are
**real demand** and deleting them would understate what the company must supply.

They are winsorised at the 99th percentile of each product's own non-zero months
for model **fitting** only; all scoring, safeguards and reported figures use
untrimmed actuals.

*(Correction note: an earlier version of this document claimed the winsorisation
while the code did not yet perform it. It was implemented and A/B-tested on
2026-08-05 together with the TSB candidate: back-test WAPE improved on both folds,
1.032 → 1.027 and 0.637 → 0.636. Controlled by `cleaning.winsorise_quantile`;
set to 0 to disable.)*

---

## 12. Discontinued products are not auto-excluded

The supplied data dictionary advises filtering out `Discontinue` products. **We did
not follow this**, because **171 discontinued products still recorded sales in
FY2024** (1.4% of volume). Dropping them would create a guaranteed shortage on
products customers are still buying.

They are forecast normally and flagged with their production status in every output.

---

## 13. Customer-level figures are an allocation, not a forecast

Individual customer × product series are far too sparse to forecast directly. Each
product's forecast is split across its customers by their share of that product over
the **last 12 months**.

Labelled prominently as an allocation everywhere it appears. Suitable for account
planning and conversations; **not** a commitment.

---

## 14. Master-list reconciliation found no problems

This was expected to be a major cleaning task. It was not:

| Check | Result |
|---|---|
| Sales items missing from item master | **0** |
| Item-master products never sold | **0** |
| Sales customers missing from customer master | **0** |
| Customer-master customers never buying | **0** |
| Item ID ↔ Item Code | strictly **1:1** |
| Customer ID ↔ Company Name | strictly **1:1** |
| Missing values in the sales file | **none** |

No fuzzy name matching or entity de-duplication was required. The checks still run
on every execution and are published in `data_quality_report.csv` as evidence — and
so they will catch problems in future uploaded data.

---

## 16. Seasonal models are locked out — measured, and accepted

**The finding.** All seven seasonal models (Holt-Winters ×4, SARIMA ×3) win **0 of
5,706 products**. This is not because they are bad — SARIMA(1,1,1)(1,0,1,12) ranked
**best of ~24 methods** on Segment A products in the fold it could run on.

**The cause.** Fold 1 has only 24 months of training data. Holt-Winters needs 25
observations and SARIMA needs 28, so both score *nothing* on Fold 1. Model selection
prefers models tested on every fold, so seasonal models lose by default.

**It was tested, not assumed.** Allowing them to compete on Fold 2 alone, with a 10%
evidence penalty, made the forecast measurably **worse**:

| | Fold 2 total error | Fold 2 WAPE | FY2025 forecast |
|---|---|---|---|
| Seasonal models excluded (current) | **−19.2%** | **0.636** | 180,238 units |
| Seasonal models allowed | −21.7% | 0.659 | 209,336 units |

**Why allowing them backfires.** The two folds differ enormously in difficulty —
Fold 1 WAPE ≈ 1.03 versus Fold 2 ≈ 0.63 for every method. A model that skips Fold 1
gets a free pass on the hard exam, so its average score flatters it relative to
models that sat both. Selecting a model using the only fold it ran on is also
circular: it is chosen and judged on the same data.

It also corrupts the reporting — the Fold 1 summary silently collapses to the biased
subset of products whose chosen model happened to run on that fold (actual units
fell from 142,929 to 61,719, which is a reporting artifact, not a data change).

**Decision: excluded.** Controlled by `backtest.allow_incomplete_fold_models`.

**The real fix, if more history arrives:** replace the two annual folds with a
rolling-origin scheme whose training windows are all ≥30 months, e.g. four 6-month
test windows stepping through FY2023–FY2024. Every model then competes on every
fold and the comparison becomes fair. With only 48 months this is not possible
without giving up the FY-aligned back-test the brief specified.

---

## 17. Prediction intervals are empirical, and must not be summed

**Method.** For each product, the selected model's back-test forecast (safeguards
included) is compared to what really sold: `ratio = actual / forecast`. Ratios are
winsorised to [0, 5] and pooled within each reliability segment; the 10th and 90th
percentiles become the low/high multipliers. Segments with fewer than 40
observations fall back to the all-product pool.

**Why empirical rather than a formula.** Demand here is intermittent, heavily skewed
and bounded below by zero. A normal-theory interval would be symmetric and could go
negative, which is meaningless. The measured distribution is none of those things —
Segment C's 10th percentile is exactly 0.00, because a lumpy product genuinely can
sell nothing in a year.

**Revised after code review: the measurement is now out-of-sample.** The original
implementation measured the ratios on the same two folds used to select the models,
which is circular and produces intervals that are systematically too narrow. The
multipliers are now computed by re-selecting models using only the earlier fold and
measuring ratios exclusively on the most recent fold, which played no part in
choosing them. The portfolio-level bracket (below) still uses both folds, since its
purpose is to span the crash-year and rebound-year regimes.

**The aggregation trap.** Adding up per-product intervals gave 74,226 – 367,146
units, so wide it carried no information. That calculation implicitly assumes all
5,706 products hit their worst case in the same year — i.e. perfect correlation of
idiosyncratic risk, which is nonsense.

Portfolio-level uncertainty is instead dominated by **systematic** risk, and we have
direct evidence of exactly that: the company total landed at **0.66×** forecast in
the crash year and **1.21×** in the rebound year. Those become the company-wide
multipliers, giving 119,835 – 219,313 units. Two observations is few, but they
bracket the two relevant regimes.

Both figures are published; the naive summed version is retained in
`run_metadata.json` as `forecast_quantity_low_if_summed` purely so the difference is
visible and auditable.

---

## 18. The pooled ML model was rebuilt to use cross-series information

**The original flaw.** The global gradient-boosting model was given only each
product's OWN lagged demand. That wasted the entire point of pooling products into
one model. It ranked **16th of ~24 methods** on the dense products — below average on
exactly the segment where it should have had the most data.

**The fix.** It now also sees, at every origin and using only data available at that
point: its product category's level, 6- and 12-month trend and year-on-year change;
its own share of that category; and the whole market's level and year-on-year change.
Category and market series are dense and well behaved even when an individual product
is not, so this is where a sparse product gets context it cannot get from its own
three data points. Categories are also now passed as native tree categoricals rather
than integer codes, and training rows carry a 24-month recency half-life.

**Result — honest.**

| | Before | After |
|---|---|---|
| Median rank, Segment A | 16th | **12th** |
| Top-3 finishes, Segment A | 11.4% | **16.0%** |
| Products won | 148 | **178** |
| Share of forecast revenue | 3.8% | **4.2%** |
| Fold 2 total error (portfolio) | −19.24% | −19.04% |

The ML component improved clearly. The **portfolio barely moved**, because ML is only
~4% of the forecast by design — and that is the architecture working as intended: the
per-product bake-off means a weak component cannot damage the result, and a stronger
one only wins where it earns it.

---

## 19. The VIP 37% discount — APPLIED (team decision 2026-08-16)

**Decision.** The discount is applied (`pricing.vip_discount: 0.37`): the data
dictionary states it, and the paired-product test below shows it is not already
inside `Base Price`, so VIP invoices are taken to be 37% below the recorded
price. Forecast revenue is therefore stated **net of the VIP discount**; the
undiscounted figure is always shown alongside it on the Data Quality page and in
`run_metadata.json` (`revenue_without_vip_discount`). The remaining
uncertainty is which basis the FY2025 grading actuals will use — if they arrive
as quantity × Base Price, the undiscounted figure is the comparable one. Setting
the switch to `0.0` restores that basis in one line.

The analysis that led here is kept in full below.

**What the dictionary says.** The `Customer` sheet defines Customer Type as:
*"VIP: 37% discount from List Price"*. The main Data Dictionary sheet repeats it:
*"VIP = Discount -37 / End User = Primary Price"*.

**Why it matters so much.** Only **10 of 2,159 customers** are VIP — but they account
for **62.7% of all quantity** and **61.9% of gross revenue**.

**What the data actually shows.** If `Base Price` already had the VIP discount applied,
VIP lines would sit near 0.63× the master list price. They do not:

| Customer type | Order lines | Median Base Price ÷ List price |
|---|---|---|
| End User | 95,336 | **0.910** |
| VIP | 170,444 | **0.891** |

VIP and End User are priced almost identically, and only **0.6%** of VIP lines sit near
0.63× list. So `Base Price` looks like a standard/reference price that does **not**
have the VIP discount applied.

**Paired-product test (added 2026-08-05, prompted by Paul).** The sharpest test
possible: for the 2,664 products bought by BOTH customer types, compare the median
Base Price paid by VIP against End User on the same product. Result: median ratio
**1.000**; 51% of shared products identical to within 2%; the top-50 volume products
identical on every single one; only 4.1% of products show anything near a 37% gap.
Base Price is therefore a **standard catalogue price attached to the transaction,
not a customer-specific price** — exactly as its own definition reads ("standard
unit selling price of the product at the time of transaction"). This is the
strongest evidence yet that the VIP discount, if real, is applied somewhere this
file does not show.

**The exposure.**

| | Forecast FY2025 revenue |
|---|---|
| As delivered (Base Price as recorded) | **THB 2,045.4m** |
| If VIP invoices really are 37% lower | THB 1,584.3m |
| **Difference** | **THB 461.1m — 22.5% of forecast revenue** |

**Why the earlier +2.3% revenue validation does NOT settle this.** That check compared
our pricing logic against "actual invoiced revenue" computed as quantity × `Base Price`
for billed rows — i.e. from the same `Base Price` field. It validates the *price
selection* logic (recent vs historical vs list), **not** whether `Base Price` equals
the amount actually invoiced. There is no invoice-amount column in the data, so no
internal check can resolve this.

**Earlier default (2026-08-04 to 2026-08-16): nothing applied**, with the sensitivity
published. Superseded by the team decision above: applied at 37%, per product in
proportion to that product's VIP sales share. (The exposure table above uses the
figures of the run current at the time it was written; the live numbers are in
`run_metadata.json`.)

**Question still worth asking Mitsubishi if the chance arises:** *"Is the Base Price
column in the sales export the price actually invoiced, or a reference price before
the VIP 37% discount is applied?"* — a "yes, invoiced" answer would mean flipping the
switch back to `0.0`.

**Note:** the quantity/demand forecast is completely unaffected. This is a revenue
question only.

---

## 20. The data conforms to the dictionary's pick lists

Checked every categorical field against the pick lists in the dictionary's `Item` and
`Customer` sheets:

| Field | Distinct in data | Listed in dictionary | Result |
|---|---|---|---|
| Production Status | 3 | 3 | conforms |
| Product Category 1 | 16 | 16 | conforms |
| Product Category 2 | 87 | 86 | conforms + `- None -` |
| Customer Type | 2 | 2 | conforms |
| FA Sub Industries | 69 | 68 | conforms + `- None -` |

The only extra value is `- None -`, which the dictionary itself explains: *"None = the
industry has not yet been specified (unknown or not assigned)"*, as distinct from
`Other` (known, but outside the classification). No invalid or misspelled categories
exist, which is why no category re-mapping was needed.

**Also captured from the `Item` sheet:** `To Be Advised` means *"scheduled for
discontinuation but remains available for sale"* — 70 such products, 54 still selling in
FY2024 (4.8% of volume). They are forecast normally and flagged, not dropped, which the
definition supports. And the `List price` note confirms our price handling: *"Some
products do not have a fixed List Price, as pricing is determined based on project
requirements and commercial conditions."*

---

## 21. Event-oriented methods and ensembling: tested, not adopted

The error diagnosis (docs/IMPROVEMENT_ROADMAP.md) showed demand is event-driven.
Three event-oriented methods and one ensemble policy were therefore built and
tested against the withheld FY2024 on 2026-08-04. None earned adoption; the full
numbers are in docs/IDEAS.md sections A and B. In brief:

| Candidate | Result |
|---|---|
| Base + spike decomposition | WAPE 0.699 vs delivered 0.635 — rejected |
| Customer-renewal (churn-aware) | WAPE 0.727 vs delivered 0.635 — rejected as forecaster; the churn signal itself is real (+3.0% vs plain averaging) and becomes an ML-feature candidate |
| Precursor analysis | Signal confirmed (1.31x lift) — validates the sales-input channel; not strong enough for statistical spike timing |
| Ensemble top-2 | −4.3% in isolation, a wash when deployed over the two-fold selection; kept behind `models.ensemble_top_k`, default off |

The consistent pattern: the existing per-product tournament is already near the
ceiling of this dataset (hindsight-best selection would recover only ~10% of
remaining error). Event-driven error requires event DATA — the sales pipeline,
customer plans — not more elaborate models of the sales history.

---

## 15. What could still go wrong

1. **A market turning point in FY2025.** Nothing in the supplied data explains why
   FY2023 collapsed 43%. With only internal order history and no external drivers,
   no method can anticipate the next turn. **This is the dominant risk.**
2. **Customer concentration.** The top 10 customers are **63.7% of all revenue**, and
   the active customer count fell from 1,246 to 869 over four years. One large
   account leaving would hurt more than any forecasting error in this document.
3. **FY2024 contained two exceptional months** (Aug 26,952 and Oct 25,612 units vs a
   ~14,100 median). The FY2025 forecast of 180,260 sits between the de-spiked run
   rate (~169,000) and FY2024's actual (191,592) — a deliberate central estimate. If
   those spikes represent a new normal, the forecast is low; if they were one-offs,
   it is about right.
4. **Segment C products are individually unreliable** by nature. They are 8% of
   volume and are included so nothing is missing from the plan, not because each
   number is precise.
