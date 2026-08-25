# Judgmental Cross-Check of the FY2025 Forecast

An independent forecast built by analyst reasoning from business facts - no
statistical fitting - as a cross-check on the model pipeline. Standard practice:
if two independent routes agree, the number is corroborated; if they disagree,
the gap itself is information. **This is a cross-check, not the deliverable.**

## Construction (units)

| Layer | Reasoning | Amount |
|---|---|---|
| Underlying run rate | FY2024 excl. order lines >=200 = 172,469; momentum flat-to-soft (H2/H1 0.984; March push 25% below seasonal norm; deseasonalised Q4 implies ~164k) -> -3% | ~167,000 |
| Event layer | Spike volume 24.8k / 5.0k / 10.1k / 19.1k across FY21-24; FY24 held one item shipping 5,000 twice; expected value with regression to mean | +13,000 |
| New-product cohort | ~550 new products appear each year selling ~2,100 first-year units (6.3k/2.1k/2.2k last three cohorts). A per-product model structurally cannot forecast products that do not yet exist. | +2,000 |
| Customer momentum | 9 of top-10 accounts grew in FY2024, most with H2 shares >50% (accelerating); churn nearly stopped (-209 in FY23, -30 in FY24); new whale account 2556 ramped 223 -> 12,828 | +3,000 |
| Discontinuation fade | 11,819 units on Discontinue/TBA products; partial fade beyond base drift | -2,000 |
| **Judgmental total** | | **~183,000** (range 155,000-205,000) |

Revenue: 183,000 x ~THB 10,700 realised average -> **~THB 1,960m**.

## Comparison

| | Judgmental | Model pipeline |
|---|---|---|
| FY2025 demand | ~183,000 | 174,514 |
| FY2025 revenue (Base-Price terms) | ~THB 1,960m | THB 1,903.8m |
| Gap | **+4.9%** (units) / +3.0% (revenue) | - |

*(Model figures refreshed 2026-08-24 after the engineered-ML-feature integration
(176,044 → 174,507) and the seasonal-reshape rounding interaction (→ 174,514).
Revenue is compared in Base-Price terms because this
judgmental build was constructed that way; the published headline is stated net of
the VIP 37% discount at THB 1,461.1m.)*

Two independent routes land within 4%. The judgmental figure sits higher chiefly
because of the new-product allowance (+2k, invisible to the model by construction)
and the customer-momentum reading (+3k, a business judgement the model does not make).

## Findings worth carrying forward

1. **New-product blind spot, quantified:** ~+2,000 units/year arrives on products
   that did not exist at forecast time. Report as a portfolio-level memo line; it
   cannot be assigned to product IDs in the CSV.
2. **The weak March:** FY2024's year-end push (15,320) ran ~25% below its seasonal
   norm (~20,400) - either softening demand or order-timing shift. One line of
   caution in the report.
3. **Customer narrative for the presentation:** churn stabilised, nine of ten top
   accounts growing, one new whale ramping - the business reading is mildly more
   optimistic than the statistics alone.

*Prepared 2026-08-05. Facts derived exclusively from the supplied data files.*
