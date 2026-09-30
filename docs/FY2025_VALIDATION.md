# FY2025 forecast versus TNIC actual demand

## Result for the final presentation

The provided actuals cover 1 April 2025 to 31 March 2026, exactly matching the
forecast horizon. Actual demand was **231,080 units**. The original committed
forecast contains **174,428.47 expected units**, a **24.52% underforecast**.
The dashboard's **174,514-unit** original headline sums per-product rounded annual
totals; it differs slightly from the fractional monthly predictions used for
accuracy calculations. Keep that rounding distinction in the presentation.

The original forecast improves product-month WAPE by **4.64% relative to the best
of the two simple baselines below**, but the annual level is too low. The flat
last-year average beats the model at the company-month level. This is evidence
of useful product-level forecasting with an unresolved portfolio-level bias,
not evidence that the model wins every decision level.

| Frozen prediction / baseline | Product-month WAPE | Product-month RMSE (units) | Product-month MAPE* | Company-month WAPE | Annual bias |
|---|---:|---:|---:|---:|---:|
| Original repository forecast | 75.67% | 26.62 | 94.48% | 24.72% | -24.52% |
| Repeat FY2024 by month | 94.13% | 39.28 | 122.20% | 31.68% | -17.09% |
| FY2024 annual volume / 12 | 79.35% | 26.69 | 99.77% | 21.49% | -17.09% |
| Setup-regenerated forecast (separate comparison) | 76.15% | 26.61 | 94.46% | 24.11% | -23.71% |

*MAPE uses 14,877 nonzero actual product-month cells and excludes 61,323 zero
actual cells. WAPE and RMSE include all 76,200 cells across 6,350 product IDs and
12 months. Do not call `100 - MAPE` the system's overall accuracy. Company-month
RMSE for the original forecast is 6,665.29 units; company-month MAPE is 20.82%.
These aggregate metrics describe a different decision from product-month errors.

## Where it missed

- June: 25,413 actual versus 15,106.74 forecast (-40.56%).
- July: 27,890 actual versus 13,654.60 forecast (-51.04%).
- January: 27,730 actual versus 15,161.60 forecast (-45.32%).
- Together these three months account for 37,110.06 units, about 65.5% of the net
  annual underforecast. These observations do not establish the cause of the spikes.
- The actual workbook's `SV` category has 63,085 actual units versus 29,652.40
  forecast: a 33,432.60-unit shortfall, about 59.0% of the net annual underforecast.
  Category labels use the supplied FY2025 item master, with the forecast master
  as fallback. No category label is used to join transactions.
- 644 products outside the forecast catalogue contributed 7,164 units (3.10% of
  actual demand). They are included in scoring with zero forecast; excluding them
  would hide a real coverage gap. This is not the main explanation for the miss.
- Products with zero annual forecast contributed 10,844 actual units, including
  those outside the catalogue. This exposes a need to review dormant-item and
  new-product planning.

## Evaluation rules and evidence

Source: `C:/Users/Doki/Downloads/TNIC Data_2025.xlsx`, sales sheet
`FY25 - Sale Order History`, 71,289 order lines. Dates are Excel serial numbers
converted using the workbook's 1900 date system. The `Item` sheet supplies
categories. Every overlapping product ID agrees with its forecast product code.

The main prediction is from Git commit
`523dc6300ee9a722bbff0e2f3c5d41de879bc950`, predating receipt of the actuals in
this chat. The setup-regenerated forecast is scored separately because its
dependencies differ from the original environment. Both predictions are saved
as snapshots, with SHA-256 hashes in the validation metadata.

The target remains demand, consistent with the original project: all order
statuses count, including Closed. The 1,726 exact duplicate lines are retained
under the existing ERP-line policy. Do not silently change those rules to improve
the score. Confirm the interpretation with the organizer if they instead expect
billed sales. Product-month absences are treated as zero within the supplied
complete-year extract; extract completeness remains a source assumption.

Baselines use only April 2024 to March 2025 history. No FY2025 actuals were used
to refit models, set multipliers, or choose predictions in this evaluation.

Revenue accuracy is not yet established. The workbook provides Base Price, not
actual invoice revenue; 1,213 lines have Base Price zero, meaning unresolved
price under the project definition. Billing statuses and VIP discount treatment
also need a consistent comparison. Do not present `Quantity × Base Price` as
verified net invoiced revenue.

## What to present and do next

1. Show a monthly forecast-versus-actual comparison, the 231,080-unit actual
   total, the original rounded headline, and the 24.52% demand underforecast.
2. Show the baseline table. Explain that the product-month improvement is 4.64%
   while the simple flat baseline is better for company-month totals.
3. Show the three spike months and the SV category as the largest areas to
   investigate. Describe order concentration, customer changes and product
   reactivation as hypotheses, not established causes.
4. Recommend a pilot with separate checks for portfolio bias, category demand,
   dormant products and new catalogue items. Track service level, stockouts,
   average inventory and forecast error.
5. Keep this FY2025 evaluation frozen. Future model changes should be tested on
   another untouched period or a chronological validation scheme. An improvement
   fitted to these revealed actuals is retrospective and must be labelled so.

Reproduce with `python validate_fy2025.py --actuals "C:/Users/Doki/Downloads/TNIC Data_2025.xlsx"`.
Detailed monthly, product, category and metric outputs are in
`outputs/fy2025_validation/`. Existing forecasts and dashboard data are unchanged.
