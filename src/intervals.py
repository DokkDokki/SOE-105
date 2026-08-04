"""Prediction intervals derived from how the pipeline ACTUALLY performed.

A single number implies a precision this data cannot support. These intervals
replace that with an honest range.

Method - empirical, not theoretical. Rather than assuming errors are normally
distributed (they are emphatically not: demand here is intermittent, skewed and
bounded below by zero), we use the errors the pipeline really made on the two
hidden back-test years:

    1. For every product, take the model that was actually selected for it, and
       the forecast it would have produced for each hidden year - safeguards
       included, so this measures what the business would really have received.
    2. Compute the ratio  actual / forecast  for that year.
    3. Pool those ratios within each reliability segment. Segments behave very
       differently: a regular seller lands close to its forecast, a lumpy one
       can easily sell nothing at all.
    4. The 10th and 90th percentiles of that pooled ratio become the low and
       high multipliers applied to the FY forecast.

So "low" and "high" mean: *in back-testing, 8 out of 10 products in this
segment landed between these two figures.* That is a statement about measured
performance, not a modelling assumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Ratios are winsorised before pooling: a product forecast at 0.2 units that
# sold 40 produces a ratio of 200, which would otherwise dominate the quantile.
RATIO_CLIP = (0.0, 5.0)
MIN_SAMPLES = 40          # below this, fall back to the all-product pool


def _annual_ratios(
    fold_store: dict, selection: pd.DataFrame, panel: pd.DataFrame, cfg: dict,
    fiscal_month_of: dict,
) -> pd.DataFrame:
    """Actual-to-forecast ratio per product per fold, using delivered forecasts."""
    from .features import seasonal_profile
    from .forecast import apply_safeguards

    sel_map = selection.set_index("item_id")["model"].to_dict()
    item_ids = panel.index.to_numpy()
    rows = []

    for fold_name, store in fold_store.items():
        preds, actual = store["preds"], store["actual"]
        h = actual.shape[1]
        chosen = np.full((len(item_ids), h), np.nan)
        for i, iid in enumerate(item_ids):
            m = sel_map.get(iid)
            if m and m in preds and not np.isnan(preds[m][i]).any():
                chosen[i] = preds[m][i]
        valid = ~np.isnan(chosen).any(axis=1)
        if valid.sum() == 0:
            continue

        train_vals = store["train_values"]
        train_panel = pd.DataFrame(train_vals, index=item_ids,
                                   columns=panel.columns[: store["split"]])
        seas = seasonal_profile(train_panel, fiscal_month_of)
        hist_fms = np.array([fiscal_month_of[c] for c in train_panel.columns])
        guarded, _, _ = apply_safeguards(
            chosen[valid].copy(), train_vals[valid], hist_fms, seas, cfg)

        a = actual[valid].sum(axis=1)
        f = guarded.sum(axis=1)
        rows.append(pd.DataFrame({
            "item_id": item_ids[valid], "fold": fold_name,
            "actual": a, "forecast": f,
        }))

    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    # Only products we actually forecast something for can produce a ratio.
    df = df[df["forecast"] > 0].copy()
    df["ratio"] = (df["actual"] / df["forecast"]).clip(*RATIO_CLIP)
    return df


def compute_multipliers(
    fold_store: dict, selection: pd.DataFrame, panel: pd.DataFrame, cfg: dict,
    fiscal_month_of: dict, profile: pd.DataFrame,
    low_q: float = 0.10, high_q: float = 0.90,
) -> tuple[pd.DataFrame, dict]:
    """Low/high multipliers per reliability segment, plus a portfolio summary."""
    ratios = _annual_ratios(fold_store, selection, panel, cfg, fiscal_month_of)
    if ratios.empty:
        return pd.DataFrame(), {}

    ratios = ratios.merge(profile["tier"].rename("tier").reset_index(),
                          on="item_id", how="left")

    overall_low = float(ratios["ratio"].quantile(low_q))
    overall_high = float(ratios["ratio"].quantile(high_q))
    overall_med = float(ratios["ratio"].median())

    out = []
    for tier, g in ratios.groupby("tier", observed=True):
        if len(g) >= MIN_SAMPLES:
            lo, hi, md, src = (float(g["ratio"].quantile(low_q)),
                               float(g["ratio"].quantile(high_q)),
                               float(g["ratio"].median()), "own segment")
        else:
            lo, hi, md, src = overall_low, overall_high, overall_med, "all products"
        out.append({
            "tier": tier, "low_multiplier": lo, "median_multiplier": md,
            "high_multiplier": hi, "products_measured": int(g["item_id"].nunique()),
            "observations": int(len(g)), "basis": src,
        })
    mult = pd.DataFrame(out)

    # ---------------------------------------------------------- portfolio level
    # The per-product bands must NOT be added up. Summing every product's 10th
    # percentile assumes all 5,706 products hit their worst case in the same
    # year, which is statistically absurd - it produced a range of 74k-367k
    # units, so wide it carried no information.
    #
    # Portfolio uncertainty is dominated by SYSTEMATIC (market-wide) risk, not
    # by individual products varying independently. We have direct evidence of
    # exactly that: how the company total landed in each hidden year. Two
    # observations is few, but they bracket a crash year and a rebound year,
    # which is the relevant risk.
    per_fold = ratios.groupby("fold", observed=True)[["actual", "forecast"]].sum()
    per_fold["ratio"] = per_fold["actual"] / per_fold["forecast"]
    fold_ratios = {k: float(v) for k, v in per_fold["ratio"].items()}

    summary = {
        "overall_low_multiplier": overall_low,
        "overall_median_multiplier": overall_med,
        "overall_high_multiplier": overall_high,
        "quantiles": [low_q, high_q],
        "portfolio_ratio_by_fold": fold_ratios,
        "portfolio_low_multiplier": min(fold_ratios.values()) if fold_ratios else np.nan,
        "portfolio_high_multiplier": max(fold_ratios.values()) if fold_ratios else np.nan,
        "portfolio_basis": (
            "The company-wide range comes from how the TOTAL actually landed in "
            "each hidden back-test year, not from adding up the per-product "
            "ranges. Per-product ranges must never be summed - that assumes "
            "every product has its worst year simultaneously."
        ),
        "n_observations": int(len(ratios)),
    }
    return mult, summary


def apply_intervals(
    forecast_long: pd.DataFrame, forecast_by_product: pd.DataFrame,
    multipliers: pd.DataFrame, profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach low/high quantity and revenue columns to the forecast tables."""
    if multipliers.empty:
        return forecast_long, forecast_by_product

    m = multipliers.set_index("tier")
    tier_of = profile["tier"]

    def _attach(df: pd.DataFrame) -> pd.DataFrame:
        t = df["item_id"].map(tier_of)
        lo = t.map(m["low_multiplier"]).astype(float)
        hi = t.map(m["high_multiplier"]).astype(float)
        # Products with no measured segment keep a symmetric fallback.
        lo = lo.fillna(float(m["low_multiplier"].mean()))
        hi = hi.fillna(float(m["high_multiplier"].mean()))
        df = df.copy()
        df["forecast_quantity_low"] = (df["forecast_quantity"] * lo).round(2)
        df["forecast_quantity_high"] = (df["forecast_quantity"] * hi).round(2)
        if "forecast_revenue" in df.columns:
            df["forecast_revenue_low"] = df["forecast_revenue"] * lo
            df["forecast_revenue_high"] = df["forecast_revenue"] * hi
        return df

    return _attach(forecast_long), _attach(forecast_by_product)
