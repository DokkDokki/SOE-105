"""Build the monthly product panel and profile every product so it can be
routed to a model class its history can actually support.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Segment names used throughout the project and shown in the dashboard.
TIER_DENSE = "A - Regular seller"
TIER_MODERATE = "B - Occasional seller"
TIER_SPARSE = "C - Rare / lumpy seller"
TIER_DORMANT = "D - No recent sales"

TIER_PLAIN_ENGLISH = {
    TIER_DENSE: "Sells almost every month. Enough history for full statistical "
                "forecasting, including seasonality.",
    TIER_MODERATE: "Sells in some months but not most. Enough history for a "
                   "simple trend, with seasonality borrowed from its product category.",
    TIER_SPARSE: "Sells only occasionally and unpredictably. Forecast as an "
                 "average demand rate, not a monthly pattern. Treat with caution.",
    TIER_DORMANT: "No sales at all in the most recent fiscal year. Forecast as "
                  "zero unless the business knows it is coming back.",
}


def build_monthly_panel(
    sales: pd.DataFrame, value_col: str = "quantity", mask_col: str | None = "is_demand"
) -> pd.DataFrame:
    """Product x month matrix of summed quantity, with true zeros filled in.

    Months in which a product sold nothing are genuine zeros, not missing data -
    that distinction is what makes intermittent-demand modelling possible.
    """
    df = sales
    if mask_col and mask_col in df.columns:
        df = df[df[mask_col]]
    if df.empty:
        return pd.DataFrame()

    panel = df.pivot_table(
        index="item_id", columns="month", values=value_col, aggfunc="sum"
    )
    full_range = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    panel = panel.reindex(columns=full_range).fillna(0.0)
    panel.columns.name = "month"
    return panel


def build_group_panel(
    sales: pd.DataFrame, items: pd.DataFrame, level: str = "category_1",
    mask_col: str | None = "is_demand",
) -> pd.DataFrame:
    """Same as build_monthly_panel but aggregated to a product-category level."""
    df = sales
    if mask_col and mask_col in df.columns:
        df = df[df[mask_col]]
    if df.empty or items.empty or level not in items.columns:
        return pd.DataFrame()

    mapping = items.set_index("item_id")[level]
    df = df.assign(_grp=df["item_id"].map(mapping).fillna("Unknown"))
    panel = df.pivot_table(index="_grp", columns="month", values="quantity", aggfunc="sum")
    full_range = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    panel = panel.reindex(columns=full_range).fillna(0.0)
    panel.index.name = level
    return panel


def profile_products(
    panel: pd.DataFrame, sales: pd.DataFrame, cfg: dict, last_fy: int | None = None
) -> pd.DataFrame:
    """One row per product describing its demand behaviour and assigned tier."""
    if panel.empty:
        return pd.DataFrame()

    seg = cfg["segmentation"]
    values = panel.to_numpy(dtype=float)
    nonzero = values > 0

    active_months = nonzero.sum(axis=1)
    n_months = values.shape[1]

    # First and last month with a sale.
    first_idx = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), -1)
    last_idx = np.where(
        nonzero.any(axis=1), n_months - 1 - nonzero[:, ::-1].argmax(axis=1), -1
    )

    months = list(panel.columns)
    prof = pd.DataFrame(index=panel.index)
    prof["active_months"] = active_months
    prof["total_months"] = n_months
    prof["first_sale_month"] = [months[i] if i >= 0 else pd.NaT for i in first_idx]
    prof["last_sale_month"] = [months[i] if i >= 0 else pd.NaT for i in last_idx]
    prof["months_since_last_sale"] = n_months - 1 - last_idx
    prof["months_of_life"] = np.where(first_idx >= 0, n_months - first_idx, 0)
    prof["total_quantity"] = values.sum(axis=1)
    prof["mean_monthly_qty"] = values.mean(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_nz = np.where(active_months > 0, values.sum(axis=1) / np.maximum(active_months, 1), 0)
        prof["mean_qty_when_selling"] = mean_nz
        # Average interval between sales - the standard intermittency measure.
        prof["avg_demand_interval"] = np.where(
            active_months > 0, prof["months_of_life"] / np.maximum(active_months, 1), np.inf
        )
        sd = values.std(axis=1)
        mu = values.mean(axis=1)
        prof["cv2"] = np.where(mu > 0, (sd / np.maximum(mu, 1e-9)) ** 2, 0.0)

    # Syntetos-Boylan demand classification (the standard scheme).
    adi = prof["avg_demand_interval"].to_numpy()
    cv2 = prof["cv2"].to_numpy()
    prof["demand_pattern"] = np.select(
        [(adi < 1.32) & (cv2 < 0.49), (adi < 1.32) & (cv2 >= 0.49),
         (adi >= 1.32) & (cv2 < 0.49)],
        ["Smooth", "Erratic", "Intermittent"], default="Lumpy",
    )

    # Recent activity, used to decide dormancy.
    if last_fy is not None and "fiscal_year" in sales.columns:
        recent = sales[sales["fiscal_year"] == last_fy]
        recent_ids = set(recent.loc[recent.get("is_demand", True), "item_id"].unique())
    else:
        recent_ids = set(panel.index[panel.iloc[:, -12:].sum(axis=1) > 0])
    prof["sold_in_last_fy"] = prof.index.isin(recent_ids)

    # ------------------------------------------------------------ tiering
    dense_min = seg["dense_min_active_months"]
    mod_min = seg["moderate_min_active_months"]
    dormant_on = seg.get("dormant_if_no_sales_in_last_fy", True)

    tier = np.where(
        dormant_on & ~prof["sold_in_last_fy"].to_numpy(), TIER_DORMANT,
        np.where(active_months >= dense_min, TIER_DENSE,
                 np.where(active_months >= mod_min, TIER_MODERATE, TIER_SPARSE)),
    )
    prof["tier"] = tier
    prof["tier_meaning"] = prof["tier"].map(TIER_PLAIN_ENGLISH)

    # Confidence label shown to non-technical users.
    prof["confidence"] = np.select(
        [prof["tier"] == TIER_DENSE, prof["tier"] == TIER_MODERATE,
         prof["tier"] == TIER_SPARSE],
        ["High", "Medium", "Low"], default="Not forecast",
    )
    return prof


def seasonal_profile(panel: pd.DataFrame, fiscal_month_of: dict, min_years: int = 2) -> np.ndarray:
    """Average within-year seasonal shape (12 factors averaging 1.0).

    Pooling seasonality across many products is far more reliable than
    estimating it from a single short series, so this is used to lend a
    seasonal shape to products that cannot estimate their own.
    """
    if panel.empty:
        return np.ones(12)

    totals = panel.sum(axis=0)
    fm = np.array([fiscal_month_of[m] for m in panel.columns])
    out = np.ones(12)
    overall = totals.to_numpy().mean()
    if overall <= 0:
        return out
    for m in range(1, 13):
        sel = totals.to_numpy()[fm == m]
        if len(sel) >= min_years:
            out[m - 1] = sel.mean() / overall
    # Normalise so the factors average exactly 1.
    if out.mean() > 0:
        out = out / out.mean()
    return out


def make_supervised_frame(
    panel: pd.DataFrame, items: pd.DataFrame, fiscal_month_of: dict,
) -> pd.DataFrame:
    """Long-format feature table for the pooled machine-learning model.

    One row per product-month with lagged demand, rolling averages, calendar
    position and product attributes. A single model trained across all products
    ("global" forecasting) learns patterns that no individual sparse series
    could support on its own.
    """
    if panel.empty:
        return pd.DataFrame()

    long = panel.stack().rename("y").reset_index()
    long.columns = ["item_id", "month", "y"]
    long = long.sort_values(["item_id", "month"])

    g = long.groupby("item_id")["y"]
    for lag in (1, 2, 3, 6, 12):
        long[f"lag_{lag}"] = g.shift(lag)
    for win in (3, 6, 12):
        long[f"roll_mean_{win}"] = g.shift(1).rolling(win, min_periods=1).mean().to_numpy()
        long[f"roll_nonzero_{win}"] = (
            g.shift(1).rolling(win, min_periods=1).apply(lambda s: (s > 0).mean(), raw=True).to_numpy()
        )
    long["expanding_mean"] = g.shift(1).expanding().mean().to_numpy()

    long["fiscal_month"] = long["month"].map(fiscal_month_of).astype(float)
    long["t"] = long.groupby("item_id").cumcount()

    if not items.empty:
        attrs = items.set_index("item_id")
        for col, name in (("category_1", "cat1"), ("category_2", "cat2"),
                          ("production_status", "prod_status")):
            if col in attrs.columns:
                long[name] = long["item_id"].map(attrs[col]).astype("category")
        if "list_price" in attrs.columns:
            long["log_list_price"] = np.log1p(
                long["item_id"].map(attrs["list_price"]).fillna(0).clip(lower=0)
            )
    return long
