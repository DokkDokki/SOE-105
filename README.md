# FY2025 Demand & Revenue Forecast — Mitsubishi Electric (Thailand)

Forecasts **monthly demand (units)** and **monthly revenue (THB)** for every product
for **FY2025 (April 2025 – March 2026)**, from four fiscal years of order history.

Built to address a specific business problem: the company suffers **both stock
surplus and stock shortage**, so it cannot supply customers just-in-time. Every
design decision below is aimed at producing numbers a planner can actually order
stock against — and at making those numbers defensible to Mitsubishi.

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python run_pipeline.py
```

```bash
streamlit run app/dashboard.py
```

The pipeline takes about **2 minutes** on a modern laptop. It reads whatever it
finds in `data/raw/` and writes everything to `outputs/`. Then the dashboard opens
in your browser. Non-technical users only ever need the dashboard.

To use different data, either drop new files into `data/raw/`, or upload them on
the dashboard's **Upload new data** page — no code changes required.

---

## What you get

| File in `outputs/` | What it is |
|---|---|
| **`FY2025_forecast_by_product_month.csv`** | **The main deliverable.** Every product × every month: quantity, unit price, revenue, price flag, method used, accuracy. |
| `FY2025_forecast_summary_by_product.csv` | One row per product — full-year totals. |
| `FY2025_forecast_by_customer.csv` | Forecast allocated to customers (see caveat below). |
| `backtest_accuracy_as_delivered.csv` | **Headline accuracy** of the complete pipeline, safeguards included. |
| `backtest_accuracy_summary.csv` | Accuracy of the selected models on each test year. |
| `backtest_accuracy_by_segment.csv` | Accuracy split by product reliability segment. |
| `backtest_accuracy_by_product.csv` | Accuracy of the chosen method, per product. |
| `backtest_all_models_scored.csv` | Every method scored on every product — the full evidence base. |
| `model_selection_by_product.csv` | Which method was chosen for each product, and why. |
| `data_quality_report.csv` | Every data issue found and what was done about it. |
| `product_profile.csv` | Demand behaviour of every product. |
| `unit_price_resolution.csv` | How each product's unit price was determined. |
| `prediction_interval_multipliers.csv` | The measured low/high multipliers behind every forecast range. |
| `run_metadata.json` | Full record of the run, including every setting used. |

---

## Headline results

| | FY2024 actual | FY2025 forecast | Likely range | Change |
|---|---|---|---|---|
| Demand | 191,592 units | **180,978 units** | 119,835 – 219,313 | −5.5% |
| Revenue (invoiced) | ฿2,024.8m | **฿2,045.4m** | ฿1,354m – ฿2,479m | +1.0% |

The range is **measured, not assumed** — see "Prediction intervals" below.

3,217 of 5,706 products carry a non-zero forecast — the rest had no sales at all in
FY2024 and are forecast as zero and clearly labelled, rather than silently dropped.

---

## The fiscal year — detected, not assumed

**FY runs April → March**, so **FY2025 = Apr 2025 – Mar 2026**. This was inferred
from the data and cross-checked three ways, not taken on trust:

1. The extract begins exactly on **1 April 2021** and ends exactly on **31 March 2025**.
2. It covers exactly **48 months = 4 complete years**, with no gaps.
3. **March is the peak month** (10.9% of annual volume vs 8.3% if flat) — the
   classic Japanese fiscal year-end push.

Detection confidence: **high**. If your organisation labels fiscal years
differently (e.g. calling this FY2026), change `forecast_fy` in `config.yaml`.

---

## How the forecast is produced

### 1. Products are profiled before any model is chosen

The catalogue is extremely sparse: the **median product sells in only 3 of 48
months**. Holt-Winters and SARIMA need at least two full seasonal cycles, so
fitting them everywhere would produce confident nonsense. Every product is
therefore routed to the class of model its history can actually support:

| Segment | Definition | Products | Share of FY2024 volume | Methods allowed |
|---|---|---|---|---|
| **A — Regular seller** | ≥ 24 active months | 932 | 82% | Everything: seasonal naïve, moving averages, drift, Holt-Winters (4 variants), SARIMA (3 orders), pooled ML |
| **B — Occasional seller** | 6–23 active months | 1,032 | 11% | Moving averages, drift, exponential smoothing, Croston/SBA, category seasonal profile |
| **C — Rare / lumpy** | 1–5 active months | 1,257 | 8% | Croston, Croston/SBA, demand-rate estimators only |
| **D — No recent sales** | nothing in FY2024 | 2,485 | 0% | Forecast zero, clearly labelled |

### 2. Every method competes; the winner is chosen by back-testing

No model is assumed. Each product is scored against every method its segment
allows, on **two** hidden years:

| Fold | Trained on | Forecasts | Weight |
|---|---|---|---|
| Fold 1 | FY2021–FY2022 | FY2023 | 0.35 |
| Fold 2 | FY2021–FY2023 | FY2024 | 0.65 |

Fold 2 is the test the brief specified. **Fold 1 was added deliberately**: Fold 2's
test year is the +25% rebound, so on its own it rewards whichever model happened to
guess "up". Requiring good performance across one crash year *and* one rebound year
is a far harder, more honest test.

The winner is then **refit on all four years** to produce FY2025.

Two guards against getting fooled by a 12-point test window:

- **Simplicity margin** — a complex model only wins if it beats the best simple
  baseline by ≥5%. Otherwise the simple method is kept.
- **No leakage** — segment assignment during back-testing uses only the training
  window, never the full history.

### 3. The selection metric is not plain accuracy

This is the single most important modelling decision in the project.

Minimising absolute error on intermittent demand drives the forecast towards the
**median** of the series — and for a product that sells in 4 months out of 12, the
median is **zero**. Selecting purely on WAPE produced a **−30% systematic
under-forecast** and silently zeroed out 1,922 products that genuinely sell.

For a company already suffering stock shortages, that is exactly the wrong error.
So the selection score adds a penalty for getting the annual total wrong:

```
score = WAPE + 1.0 × |forecast_total − actual_total| / actual_total
```

A zero forecast for a product that does sell now scores 1.0 + 1.0 = 2.0 and can no
longer win by default. This cut the systematic bias from −30% to −19%.

### 3b. The machine-learning component, and why it is deliberately minor

> **No external data is used anywhere in this project.** Every figure comes from the
> three supplied files (`Sale Order History.xlsx`, `Item.xlsx`, `Customer.xlsx`).
> There are no internet calls, no market indices, no third-party or benchmark data.
> The code contains no networking libraries at all — this is verifiable by searching
> the source for `requests`, `urllib` or any URL.

The ML model ([`src/global_ml.py`](src/global_ml.py)) is a **pooled** gradient-boosting
regressor — one model fitted across all 5,706 products, rather than 5,706 separate
models, because the median product has only 3 usable observations.

*(In forecasting literature this is called a "global" model, meaning one model across
all the series in the dataset, as opposed to a "local" model fitted per series. The
word refers to the modelling structure, not to global/world data. In the dashboard it
is labelled "Machine learning (learns from all products in this dataset together)" to
avoid any confusion.)* Strategy is **direct multi-horizon**
(12 separate regressors, one per month ahead) to avoid error accumulation, with
Poisson loss so it targets the conditional *mean* rather than the median.

Crucially it uses **cross-series features**: each product sees its category's level,
trend and year-on-year change, its own share of that category, and the whole market's
level and trend — all computed using only data available at that point in time.
Category and market series are dense even when an individual product is not, which is
where a sparse product gets context its own history cannot provide. Adding these
moved it from 16th to **12th** of ~24 methods on the dense segment.

It wins **178 of 5,706 products (4.2% of forecast revenue)**. That is the right
outcome, not a failure: with 48 monthly observations, classical and intermittent-demand
methods are simply better suited, and the per-product bake-off means a weak component
cannot damage the forecast — it only wins where it genuinely earns it. Deep learning
(N-BEATS, DeepAR, LSTM) was deliberately **not** used; those need far more history
than exists here and would overfit spectacularly.

### 4. Two business safeguards

- **Shortage floor** — no product that demonstrably sold in the last 12 months is
  forecast below **25% of its own last-year volume**. The reference is last year's
  volume, deliberately not a lifetime average (a lifetime average turned a product
  selling 1 unit/year into a forecast of 24, because it had been a fast mover three
  years earlier). An earlier version triggered only on exactly-zero forecasts, which
  let a product selling 1,786 units/year be forecast at 12; a code review caught
  this and the floor is now proportional. Applied to **1,276 products**.
- **Surplus cap** — no product is forecast above **1.5× its largest annual volume in
  the last three years**. "Repeat last month" and straight-line trend can take one
  unusually large final month and project it across twelve — one product jumped
  from 290 units to 948 that way. Applied to **119 products**.

Both are reported in the dashboard and can be switched off in `config.yaml`.

### 5. Hierarchical reconciliation — tested, then rejected

Product forecasts can be scaled so they sum to independently-forecast category
totals. This was **implemented and back-tested**, and it made accuracy *worse*
(combined score 1.161 as-is vs 1.464 scaled), so it is **switched off**. The
decision and its evidence are written to `outputs/reconciliation_decision.json`
rather than being a matter of taste. If future data changes the verdict, the
pipeline will switch it on automatically.

---

## Prediction intervals — measured, not assumed

Every forecast ships as **low / best / high** rather than a single number, because a
point estimate implies a precision this data cannot support.

The intervals are **empirical**. Rather than assuming errors are normally distributed
(they are emphatically not — demand here is intermittent, skewed, and bounded below
by zero), they come from the errors the pipeline really made on the two hidden years:

1. For every product, take the model actually selected for it and the forecast it
   would have produced for each hidden year — **safeguards included**, so this
   measures what the business would really have received.
2. Compute `actual / forecast` for that year.
3. Pool those ratios within each reliability segment.
4. The 10th and 90th percentiles become the low and high multipliers.

| Segment | Low | Typical | High | Observations |
|---|---|---|---|---|
| A — Regular seller | 0.47× | 0.97× | 1.71× | 1,864 |
| B — Occasional seller | 0.20× | 1.00× | 4.00× | 1,828 |
| C — Rare / lumpy | 0.00× | 1.00× | 3.00× | 553 |

So "low to high" means: *in back-testing, 8 out of 10 products in this segment landed
between these figures.* Segment C's low multiplier of 0.00 is not a bug — a lumpy
product genuinely can sell nothing in a given year.

**Per-product ranges must never be added up.** Summing every product's 10th
percentile assumes all 5,706 have their worst year simultaneously, which produced a
meaningless 74,226 – 367,146 band. Portfolio uncertainty is dominated by *systematic*
(market-wide) risk, so the company-wide range instead comes from how the **total**
actually landed in each hidden year (0.66× in the crash year, 1.21× in the rebound
year). This distinction is enforced in code and explained in the dashboard.

---

## How revenue is calculated

```
revenue = forecast quantity × historical billed share × unit price
```

**Price = 0 is never treated as ฿0 revenue.** Per the business, a zero price means
the price is requested from the Japan branch per order and varies. Resolution
cascade, first rule that succeeds:

| # | Rule | Products |
|---|---|---|
| 1 | Quantity-weighted average price actually achieved in the last 12 months | 3,146 |
| 2 | Same, over the product's whole history | 2,480 |
| 3 | Master list price × the catalogue's realised/list ratio (0.885, computed from the data) | 75 |
| 4 | **`PRICE_TBD`** — revenue left blank, never guessed. Quantity still forecast. | **5** |

Of 1,080 products that have ever been priced at zero, **1,000 had a real price
recoverable from their own sales history**. Only 5 products (0.01% of volume) are
genuinely unknown, and they are flagged distinctly in both the CSV and dashboard.

Products whose history contains *any* zero price are flagged
`PRICE_VARIABLE_ESTIMATED` even when a price was recovered, so nobody mistakes an
estimate for a firm price.

**The pricing method is validated separately.** Running FY2024's *actual*
quantities through exactly these pricing rules reproduces FY2024's actual invoiced
revenue to within **+2.3%** — so the revenue figures rest on sound prices, and any
remaining error is forecasting error, not pricing error.

---

## Accuracy — reported honestly

| Fold | Test year | Actual | Forecast | Total error | WAPE |
|---|---|---|---|---|---|
| Fold 1 | FY2023 | 142,929 | 214,023 | **+49.7%** | 1.055 |
| Fold 2 | FY2024 | 190,094 | 153,526 | **−19.2%** | 0.636 |

**These look bad, and they are not hidden.** FY2023 is the year demand collapsed
43%; FY2024 is the year it rebounded 25%. Neither turning point was predictable
from the data preceding it — no method in the bake-off caught either. **These are
the realistic worst case, not the expected error in a normal year.** Per-product
median MASE of 0.27 on Fold 2 means the chosen methods are roughly **4× better than
a naïve "same as last year" guess** at the product level.

### A note on MAPE

MAPE is reported because it was requested, but it is **not** the selection metric.
It is mathematically undefined whenever an actual is zero — which, with a median of
3 active months out of 48, is most of this catalogue. Every MAPE figure is published
alongside the count of months it had to skip. **WAPE, MASE and total-error % are the
metrics to judge this forecast by.**

---

## Data quality — what was found and done

The two master lists reconcile **perfectly**: zero orphan records in either
direction, and Item ID↔Code and Customer ID↔Name are strictly 1:1. No fuzzy name
matching was needed. The checks still run and are published as evidence.

| Finding | Decision |
|---|---|
| **6,865 exact duplicate order lines** (1.8% of quantity) | **Kept.** Forensic evidence says they are genuine repeated ERP lines: 2,568 of 2,674 affected orders duplicate only *some* lines (an export bug would clone whole orders uniformly); they are spread evenly across all four years (3.8–5.2%); 65% carry qty 1. Toggle in `config.yaml`. |
| **`Closed` status defined two ways** in the supplied data dictionary — "completed and closed" vs "will not be fulfilled or billed" (14.5% of FY2024 volume) | **Demand counts every status** (a closed order still expressed real customer demand — the signal inventory planning needs). **Revenue counts only invoiced statuses.** Both series are published. |
| 52 rows with quantity 0 | Dropped — not transactions, zero revenue impact. |
| Large orders (max 6,000 units vs median 1) | **Never deleted.** They are real demand. Winsorised for model *fitting* only; all scoring and reporting use untrimmed actuals. |
| 171 discontinued products still selling in FY2024 (1.4% of volume) | **Not auto-excluded** despite the data dictionary's advice. Forecast and flagged instead of dropped. |
| 13 null list prices in the item master | Handled by the price cascade above. |
| Missing values in the sales file | **None.** |

---

## Known limitations

1. **The customer view is an allocation, not a forecast.** Individual customers
   order far too irregularly to forecast directly. Each product's forecast is split
   across its customers by their share of that product over the last 12 months.
   Labelled as such wherever it appears. Use for account planning, not commitments.
2. **Segment C and D products are individually unreliable.** They are included so
   nothing is missing from the plan, not because each number is precise. Together
   they are 8% of volume.
3. **No external drivers.** Nothing in the supplied data explains *why* FY2023
   collapsed. With only internal order history, a genuine market turning point in
   FY2025 cannot be anticipated — this is the dominant risk in the forecast.
4. **Customer count keeps falling** (1,246 → 869 over four years) while volume
   recovers. Revenue is concentrating into fewer accounts — the top 10 customers are
   63.7% of all revenue. A single large account leaving would hurt more than any
   forecasting error.

---

## Project structure

```
mitsubishi-forecast/
├── config.yaml               # every assumption, documented and changeable
├── requirements.txt
├── run_pipeline.py           # command-line entry point
├── data/
│   ├── raw/                  # put source files here (.xlsx or .csv)
│   └── uploads/              # files uploaded via the dashboard
├── src/
│   ├── config.py             # config loading; no absolute paths anywhere
│   ├── ingest.py             # CSV/Excel reading, tolerant column mapping
│   ├── clean.py              # cleaning, master-list reconciliation, quality report
│   ├── fiscal.py             # fiscal-year detection and calendar helpers
│   ├── features.py           # monthly panel, product profiling, segmentation
│   ├── models.py             # all candidate forecasting methods
│   ├── global_ml.py          # pooled gradient-boosting model
│   ├── metrics.py            # WAPE, MAE, RMSE, MASE, MAPE, SMAPE, bias
│   ├── backtest.py           # two-fold back-test and model selection
│   ├── forecast.py           # refit, safeguards, reconciliation
│   ├── pricing.py            # price resolution and revenue
│   └── pipeline.py           # orchestrates everything
├── app/
│   └── dashboard.py          # Streamlit dashboard
├── docs/
│   └── ASSUMPTIONS.md        # every decision, with the evidence behind it
└── outputs/                  # all generated files
```

---

## Why Streamlit

The brief needs *Python pipeline + upload a CSV + easy for everyone*. Streamlit is
the right fit because it runs **the same Python objects the pipeline produces** with
no separate API, database or JavaScript layer — so the website and the command line
can never disagree. `st.file_uploader` delivers the entire "upload new data and
re-run" requirement in one component, and the result is a plain web page any team
member can read without training. Dash would need callback plumbing; Power BI or
Tableau would need the modelling to live outside Python, breaking reproducibility.

---

## Using this in future years

The pipeline is built to be re-run, not rebuilt. Two things to know.

### 1. Refreshing the forecast — no code, no analyst

Open the dashboard → **Upload new data** → choose the newer file(s) → **Run**.
Everything downstream re-runs from scratch:

`read → validate schema → clean → reconcile against masters → profile every product →
re-segment → back-test every candidate method → re-select the winner per product →
refit on all history → apply safeguards → resolve prices → recompute revenue →
recompute prediction intervals → write all CSVs → refresh every dashboard page`

Roughly 2 minutes. The upload page shows exactly how each of your columns was matched
**before** anything runs, and refuses files missing an essential column. Only the
sales file is mandatory — the item and customer masters are reused from `data/raw/`
if you don't replace them.

The model choice is **re-decided from scratch** every run. A product that used a
6-month average this year may switch to Croston next year if that back-tests better.
That is intended: the method should follow the data, not be frozen at first build.

### 2. Switch the folds to `auto` after submission

The delivered `config.yaml` hard-codes the two back-test folds so the numbers match
this README and the brief exactly. **For ongoing use, change it to:**

```yaml
backtest:
  folds: auto
  fold_weights: []
```

Otherwise the pipeline keeps testing itself against FY2023 and FY2024 forever — going
stale while still appearing to work. With `auto`, it always tests the most recent
complete fiscal years present in the data.

**This also fixes a known limitation by itself.** Holt-Winters and SARIMA need 25–28
months of training data; today's earliest fold has only 24, so all seven seasonal
models are excluded (section 16 of ASSUMPTIONS.md). Verified behaviour as history
grows:

| History available | Folds generated | Seasonal models |
|---|---|---|
| 48 months (today) | test FY2023 (24mo train), FY2024 (36mo) | excluded — fold 1 too short |
| **60 months (once FY2025 lands)** | test FY2024 (36mo), FY2025 (48mo) | **both folds usable — unlock automatically** |
| 72 months | test FY2025 (48mo), FY2026 (60mo) | usable |

So next year the method set genuinely widens with no code change.

### 3. When the real FY2025 numbers arrive

That is the first chance to validate this work **for real** rather than by
back-testing. Add the FY2025 actuals to `data/raw/`, re-run, and the pipeline will
test itself against them automatically (with `folds: auto`). Compare against the
range published here — 119,835 – 219,313 units, best estimate 180,978 — and record
whether the outturn fell inside it. That single check is worth more than any further
modelling.

### Suggested operating rhythm

| When | Do |
|---|---|
| Monthly | Re-run with the latest closed orders. Watch the Segment A products — 82% of volume. |
| Quarterly | Review the Data Quality page; re-check the price-TBD list with the Japan branch. |
| Annually | Re-read `docs/ASSUMPTIONS.md` and confirm each decision still holds. |
| Always | Plan against the **range**, not the point. Use Segment A/B/C to decide how much to trust each number. |

---

## Reproducibility

- No absolute paths anywhere — everything resolves relative to the project root.
- Every assumption lives in `config.yaml`, not in code.
- Random seeds fixed (`random_state=42`) on the ML model.
- `run_metadata.json` records the complete configuration used for every run, so any
  number can be traced back to the settings that produced it.
- The dashboard's re-run button calls the **same** `run_pipeline()` function as the
  command line.
