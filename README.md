# Demand Forecasting & Inventory Planning — Mitsubishi Electric (Thailand)

A system that forecasts **monthly demand and revenue for every product** for
**FY2025 (April 2025 – March 2026)** and turns that forecast into a **stock plan**,
built from four fiscal years of order history.

The business problem it answers: the company has **stock surplus and stock shortage
at the same time**, so it cannot supply customers just-in-time. Too much of what does
not sell, too little of what does.

| | FY2024 actual | **FY2025 forecast** | Likely range |
|---|---|---|---|
| Demand | 191,592 units | **174,514 units** (−8.9%) | 118,093 – 215,741 |
| Revenue, net of VIP discount | — | **฿1,461.1m** | ฿988.7m – ฿1,806.3m |
| Revenue, Base-Price terms | ฿2,024.8m | ฿1,903.8m (−6.0%) | — |

Two revenue lines because the history is recorded at Base Price while VIP customers
pay 37% less; see [Revenue](#revenue-and-the-two-revenue-numbers). Every number
comes from the three files Mitsubishi supplied, and every modelling choice was
tested on years the models never saw.

**Run it** below, then read on for how the forecast is made, how accurate it is,
and how it becomes a stock plan.

---

## Run it

You need Python 3.10 or newer (built and tested on 3.13, Windows 11).

```bash
pip install -r requirements.txt
```

```bash
python run_pipeline.py
```

```bash
streamlit run app/dashboard.py
```

The pipeline reads `data/raw/`, writes everything to `outputs/`, and takes
**7–16 minutes** depending on the laptop. The dashboard then opens at
`http://localhost:8501` and loads in seconds, because it only reads the saved results.

Progress prints one model family at a time ("Back-testing models (12 fitted)…").
That is normal: each family runs across all 5,706 products in parallel underneath.
The machine-learning step is visibly the slowest.

A finished run ends with a summary. **Check these two lines before doing anything else:**

```
Forecast quantity        : 174,514 units
Forecast revenue         : THB 1,461,124,134
```

---

## How the forecast is made

### Products are sorted before any model is chosen

The catalogue is extremely sparse, so each product is only offered the methods its
history can support.

| Segment | Definition | Products | Share of FY2024 volume |
|---|---|---|---|
| **A — Regular seller** | 24+ active months | 932 | 81.6% |
| **B — Occasional seller** | 6–23 active months | 1,032 | 10.7% |
| **C — Rare / lumpy** | 1–5 active months | 1,257 | 7.6% |
| **D — No recent sales** | nothing sold in FY2024 | 2,485 | 0% |

### A tournament, scored on two hidden years

About 24 methods compete on every product: moving averages and other baselines,
exponential smoothing, the intermittent-demand specialists (Croston, Croston/SBA,
TSB), and one pooled machine-learning model.

| Fold | Trained on | Forecasts blind | Weight |
|---|---|---|---|
| 1 | FY2021–FY2022 | FY2023 (the crash) | 0.35 |
| 2 | FY2021–FY2023 | FY2024 (the rebound) | 0.65 |

Each product keeps the method with the best weighted score, and the winners are then
**refit on all 48 months** to produce FY2025. Two folds rather than one, because
selecting on FY2024 alone was measured to be **40% worse on an unseen year**
(WAPE 1.472 against 1.050). A complex method must also beat the best simple one by
5% to win.

Who won, across 5,706 products:

| Method | Products |
|---|---|
| Forecast zero | 3,394 |
| Average of last 6 / 3 / 12 months | 607 / 250 / 173 |
| Machine learning (pooled gradient boosting) | 242 |
| Croston/SBA, Croston, TSB | 214 / 67 / 61 |
| Straight-line trend | 194 |
| Recent level × category seasonal pattern | 96 |
| Everything else (8 methods) | 408 |

"Forecast zero" is the 2,485 dormant products plus about 900 rare ones whose best
back-tested forecast was zero. A safeguard then lifts any that sold in the last 12
months, which leaves **2,741 products with a non-zero forecast**.

### The selection score is not plain accuracy

Minimising absolute error on intermittent demand pulls the forecast toward the
*median*, and for a product selling 4 months in 12 the median is zero. Selecting on
error alone produced a −30% systematic under-forecast and zeroed out 1,922 real
products. For a company with shortages that is the wrong mistake to make, so the
score penalises missing the total as well:

```
score = WAPE + |forecast total − actual total| / actual total
```

### The machine-learning model

One gradient-boosted tree model (scikit-learn `HistGradientBoosting`, Poisson loss)
trained across **all products together**, because the median product has too little
history to train on alone. It uses 49 features: the product's own lags and rolling
averages, its category and the whole market, seasonality, spike history, and
customer signals such as buyer count, concentration and order frequency.

Forecasting literature calls this a "global" model, meaning one model across all
the series in this dataset. It does not mean global or external data.

### Safeguards and shaping

- **Shortage floor.** No product that sold in the last 12 months is forecast below
  25% of its own last-year volume. Applied to 1,266 products.
- **Surplus cap.** No product is forecast above 1.5× its largest annual volume of
  the last three years. Applied to 129 products.
- **Seasonal reshape.** Methods like averages and Croston produce a flat line, which
  made the year look flat (peak-to-trough 1.10 against 1.77 in history). Those
  products now get the pooled seasonal shape at half strength (1.38). Annual totals
  are untouched. Full strength lost on Fold 1 and was rejected.

---

## Accuracy, reported honestly

| Fold | Test year | Actual | Forecast | Total error | WAPE |
|---|---|---|---|---|---|
| 1 | FY2023 | 142,911 | 208,022 | **+45.6%** | 1.021 |
| 2 | FY2024 | 190,100 | 150,869 | **−20.6%** | 0.622 |

These look bad and they are not hidden. FY2023 is the year demand collapsed 43% and
FY2024 the year it rebounded 25%. No method caught either turn from the sales
history alone. They are the realistic worst case, not the error of a normal year.

**Against simple planning**, same products, same years (WAPE, lower is better):

| | Repeat last year | Last year spread evenly | This system | vs best simple plan |
|---|---|---|---|---|
| FY2023 | 1.426 | 1.243 | **1.021** | **−17.9%** |
| FY2024 | 0.845 | 0.692 | **0.622** | **−10.1%** |

**Accuracy depends on the question** (FY2024 test year):

| Decision | Accuracy |
|---|---|
| One product, one month | 37.8% |
| One product, a quarter | 54.5% |
| One product, the year | 72.3% |
| A category, one month | 74.1% |
| The company, the year | **79.4%** |

**The ceiling was measured.** With perfect hindsight, picking the best method for
every product would remove only about **10%** of the remaining error. The rest is
information that is not in the sales history. 25 ideas were tested on the hidden
years; 5 were adopted and the rest are documented in
[`docs/IDEAS.md`](docs/IDEAS.md) so nobody repeats them.

MAPE is reported because it was requested, but it is undefined whenever an actual
is zero, which is most of this catalogue. Judge the forecast on WAPE, MASE and total error.

---

## Inventory planning

The forecast is what we expect. How much to stock is a choice, so the output is a
**curve**: for each stocking policy, how much of the demand that really arrived in
FY2024 would have been served, and how much stock that meant holding.

| Target service | Average stock held | Policy |
|---|---|---|
| 70% | 8,900 units | 1.50× forecast |
| 80% | 19,700 units | 1.75× forecast, 0.50× P90 buffer |
| 85% | 29,100 units | 2.00× forecast, 0.75× P90 buffer |
| 88% | 41,000 units | 2.50× forecast, 1.00× P90 buffer |
| **90%** (default) | **74,100 units** | 0.75× forecast, 1.75× P90 buffer |
| 92% (the maximum reached) | 135,700 units | 0.75× forecast, 3.00× P90 buffer |

**The stocking rule.** Hold, per product per month, the *larger* of the forecast
times a multiplier and a multiple of that product's own P90 single-month demand.
Whole units, rounded up.

**Why a buffer.** A multiplier alone reaches 90% only at **15× the forecast, about
187,000 units held**. Lumpy demand is not under-forecast, it is differently shaped:
a product expected at 0.3 a month that occasionally takes one order of 50 needs a
150× multiplier, and a multiplier spends that on every other product as well. The
buffer reaches the same 90% with about 74,000 units, which is roughly 2.6 stock
turns a year.

**Limits.**
- The curve ends near 92%. About 2% of demand lands on products forecast at zero,
  which no stock policy can serve.
- Above about 90%, stock is set by the buffer rather than the forecast, and a naive
  forecast needs the same amount. The forecasting earns its keep between 70% and
  88%: at 88% it needs 41,000 units where a naive forecast needs 49,400.
- 88% → 90% nearly doubles the stock. The page shows that so the business can choose.

**The downloaded plan** is in whole units. Plain rounding would delete slow movers
(a product expected at 0.33 a month rounds to zero twelve times), so each product's
units are placed in its strongest months and its annual total is preserved. The file
adds up to exactly 174,514, and it uses the same stocking rule the curve was
simulated with, so the service level on the page is the one the file delivers.

A row can show *forecast 0* with stock held. Those are products expected to sell
under half a unit a year; they did sell in the test year, and their buffers are
cheaper per point of service than the rest of the curve.

---

## Shock Simulator

Applies a demand shock to the FY2025 plan and runs the same stock simulation: stock
held to the *plan*, demand arriving *shocked*. Scenario sizes come from documented
events, and nothing on the page changes a published number.

| Scenario | Stocked at 1.0× the plan |
|---|---|
| Capital-expenditure rebound, **+25%** (what FY2024 did) | 83.0% service; **37,000 units missed, about ฿310m** |
| FA market continuation, **+10%** | 93.3% service; 12,800 units missed, about ฿108m |
| Inventory-correction crash, **−43%** (what FY2023 did) | everything served; **about 8,700 units of surplus** |

A custom shock size and an onset month are available, and a per-product table shows
which products carry the damage, with a CSV download.

---

## External context

Nothing in the supplied files explains the FY2023 collapse or the FY2024 rebound,
so we went and found out. [`docs/EXTERNAL_EVENTS.md`](docs/EXTERNAL_EVENTS.md)
is an analyst overlay built from public sources. **It is a document, not an input:**
nothing in it is fitted, joined, or fed to a model.

- **FY2023, −42.8%.** The global electronics inventory correction. Mitsubishi
  Electric's own Factory Automation orders fell 43% year on year in the same window.
- **FY2024, +25.1%.** The FA capital-expenditure recovery. Not Thai automotive,
  which fell 20% that same year; the obvious story is the wrong one.
- Read at each forecast origin, FA order growth called both turns correctly, each
  time against the model. That is two observations, so it is evidence, not proof.

**Decision:** the forecast stays **174,514**. A market cross-check of
**211,000 units / ฿1,766.7m** (FY2024 actual × the FA market's +10%) is published
beside it as a labelled scenario. Both sit inside the published range. It is not
fitted as a feature because four years and two folds cannot test an annual indicator.

---

## Revenue, and the two revenue numbers

```
revenue = forecast quantity × historical billed share × unit price
```

**A price of 0 is never ฿0.** It means the price is quoted per order by the Japan
branch. Prices are resolved in this order:

| # | Rule | Products |
|---|---|---|
| 1 | Average price actually charged, last 12 months | 3,146 |
| 2 | Same, over the product's whole history | 2,480 |
| 3 | Master list price × the typical realised/list ratio (0.91) | 75 |
| 4 | `PRICE_TBD`: revenue left blank, quantity still forecast | **5** |

1,080 products have at some point been priced at zero, and almost all had a real
price recoverable from their own billed lines. The 5 left over carry about 2 units.

**The VIP discount.** The data dictionary says VIP customers pay 37% below Base
Price, and the Base Price column does not already include it. The team decision
(`pricing.vip_discount: 0.37`) is to state revenue net of it, **฿1,461.1m**, and
always show the Base-Price figure, **฿1,903.8m**, beside it. If Mitsubishi confirms
Base Price is the invoiced amount, set it to `0.0`.

Compare like with like: FY2024's ฿2,024.8m is in Base-Price terms, so the fair
comparison is ฿1,903.8m (−6.0%), not ฿1,461.1m. The pricing rules were validated
separately: FY2024's actual quantities through these rules reproduce FY2024's
invoiced revenue within +2.3% in Base-Price terms.

---

## Data decisions

Full evidence for each is in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

| Finding | Decision |
|---|---|
| **Fiscal year** | April → March, detected from the data: the extract runs exactly 1 Apr 2021 – 31 Mar 2025, 48 months with no gaps, and March is the peak month. |
| **6,863 exactly duplicated order lines** (1.8% of volume) | **Kept.** Four forensic checks show genuine ERP behaviour. Removing them moves the forecast under 2%. |
| **`Closed` orders** (14.5% of FY2024 quantity) | Counted as **demand** (the customer wanted it) but not as **revenue** (it was never billed). |
| **Spikes** | Never removed from actuals. Capped at each product's own p99 while fitting only. |
| **Monthly forecasts** | Kept fractional. They are expected demand, and rounding them deletes slow movers. The stock plan is where whole units belong. |

`clean_data.py` in the project root is a teammate's standalone data-validation
script (keys, nulls, types) that writes cleaned CSVs to a separate folder. It is not
part of the forecast pipeline.

---

## Known limitations

1. **Turning points.** With only internal order history, no method here can
   anticipate the next market turn. This is the dominant risk, and the external
   evidence says the risk on this forecast is to the upside.
2. **Customer concentration.** The top 10 customers are 68.5% of revenue (67.6% of
   units), and active customers fell from 1,246 to 869 in four years. One large
   account leaving would matter more than any forecasting error.
3. **The customer view is an allocation**, not a forecast: each product's forecast
   split by customers' recent share. Use it for account planning, not commitments.
4. **Segments C and D are individually unreliable.** They are in the plan so nothing
   is missing, not because each number is precise. Together they are under 8% of volume.
5. **Sales are censored by stock.** In a shortage, lost sales were never recorded, so
   history understates true demand exactly where it matters.

What would actually move accuracy is information, not algorithms: quotation data
(the large orders were known to sales before they landed), a 30-minute monthly
sales review of the top ~50 products, and the FY2025 actuals.

---

## What is in `outputs/`

| File | What it is |
|---|---|
| **`FY2025_forecast_by_product_month.csv`** | **The main deliverable.** Every product × month: quantity, range, price, revenue, method, accuracy. |
| `FY2025_forecast_summary_by_product.csv` | One row per product: annual total, method, P90 event exposure. |
| `FY2025_forecast_by_customer.csv` | The forecast allocated to customers. |
| `inventory_service_frontier.csv` | The service-level / stock-held curve, with the naive comparison. |
| `backtest_accuracy_as_delivered.csv` | Headline accuracy of the full pipeline, safeguards included. |
| `benchmark_comparison.csv` | This system against the two naive plans. |
| `accuracy_by_decision_level.csv` | The same forecast scored at every decision level. |
| `backtest_all_models_scored.csv` | Every method on every product. The full evidence base. |
| `model_selection_by_product.csv` | Which method each product got, and why. |
| `unit_price_resolution.csv` | How each product's price was determined. |
| `data_quality_report.csv` | Every data issue found and what was done. |
| `prediction_interval_multipliers.csv` | The measured multipliers behind every range. |
| `run_metadata.json` | Full record of the run, including every setting. |

The recommended stock plan is downloaded from the Inventory Planning page, since it
depends on the service level chosen there.

---

## Project structure

```
mitsubishi-forecast/
├── config.yaml             every assumption, documented and changeable
├── run_pipeline.py         command-line entry point  (--quick for a fast refresh)
├── requirements.txt
├── data/
│   ├── raw/                the three source files
│   └── uploads/            kept copies of files uploaded via the dashboard (never read back)
├── src/
│   ├── ingest.py           CSV/Excel reading with tolerant column matching
│   ├── clean.py            cleaning, master-list reconciliation, quality report
│   ├── fiscal.py           fiscal-year detection
│   ├── features.py         monthly panel, product profiling, segmentation
│   ├── models.py           the statistical forecasting methods
│   ├── global_ml.py        the pooled machine-learning model and its features
│   ├── backtest.py         two-fold back-test, selection, benchmarks
│   ├── forecast.py         refit, safeguards, seasonal reshape
│   ├── intervals.py        measured prediction ranges
│   ├── pricing.py          price resolution, VIP discount, revenue
│   ├── inventory.py        stocking policy, service curve, whole-unit plan
│   ├── metrics.py          WAPE, MASE, MAE, MAPE, bias
│   └── pipeline.py         runs everything in order
├── app/dashboard.py        the Streamlit dashboard
├── docs/
│   ├── ASSUMPTIONS.md      every decision, with its evidence
│   ├── IDEAS.md            every idea tested: adopted, rejected, and why
│   ├── EXTERNAL_EVENTS.md  the market-context overlay
│   ├── JUDGMENTAL_FORECAST.md   an independent analyst cross-check
│   └── IMPROVEMENT_ROADMAP.md   where further accuracy would come from
└── outputs/                all generated files
```

---

## Using it after handover

**Monthly.** Open the dashboard → *Data Upload & Refresh* → choose the newer sales
file → *Generate forecast*. The page shows how every column was matched before
anything runs and refuses a file missing an essential column. Only the sales file is
required; the item and customer masters are reused. An upload drives that one run.
To make the new file the permanent input, place it in `data/raw/`.

**Quick refresh.** Ticking *Quick refresh* (or `python run_pipeline.py --quick`)
reuses the model selection and accuracy evidence from the last full run and refits on
the new data, in under a minute instead of 7–16. Runs are labelled as quick in the
sidebar. Do a full run each quarter, and always before presenting numbers.

**After FY2025 actuals arrive**, set the back-test to follow the data:

```yaml
backtest:
  folds: auto
  fold_weights: []
```

With 60 months of history both folds become long enough for Holt-Winters and SARIMA,
which are built in but locked out today because the earliest fold has only 24
training months. They unlock with no code change. That run is also the first real
validation: record whether the outturn fell inside **118,093 – 215,741**.

| When | Do |
|---|---|
| Monthly | Refresh with the latest orders. Watch Segment A, 82% of volume. |
| Quarterly | Full run. Re-check the price-TBD list with the Japan branch. |
| Annually | Re-read `docs/ASSUMPTIONS.md` and confirm each decision still holds. |
| Always | Plan against the **range**, not the point. |

---

## Reproducibility

- No absolute paths. Everything resolves relative to the project root.
- Every assumption lives in `config.yaml`, not in code.
- The ML model uses a fixed seed (`random_state=42`); the statistical methods are deterministic.
- `run_metadata.json` records the full configuration of every run.
- The dashboard's run button calls the same `run_pipeline()` as the command line.
- No networking library is imported anywhere in `src/`.
