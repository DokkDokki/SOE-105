"""Unit-price resolution and revenue forecasting.

The central rule: a Base Price of 0 is NOT a price of zero baht. Per the
business it means the price is requested from the Japanese branch for each
order and varies. Treating it as zero revenue would understate sales.

Resolution cascade, first rule that succeeds wins:
  1. realised        - median non-zero price actually achieved for that product
                       in the last N months
  2. realised_all    - median non-zero price over the product's whole history
  3. list_adjusted   - master list price x the catalogue-wide realised/list ratio
                       (list price is a sticker price; this discounts it to what
                       is actually achieved)
  4. list_raw        - master list price as-is, if no ratio can be computed
  5. PRICE_TBD       - genuinely unknown. Quantity is still forecast; revenue is
                       left blank and flagged, never set to zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_SOURCE_PLAIN = {
    "realised": "Average price actually charged in the last 12 months",
    "realised_all": "Average price actually charged (full history)",
    "list_adjusted": "Master list price, adjusted to the typical realised discount",
    "list_raw": "Master list price",
    "PRICE_TBD": "Price varies - quoted by the Japan branch per order",
}


def realised_to_list_ratio(sales: pd.DataFrame, items: pd.DataFrame) -> float:
    """Catalogue-wide median of (achieved price / list price).

    Used to discount a sticker list price down to what is really achieved.
    """
    if items.empty or "list_price" not in items.columns:
        return np.nan
    paid = sales[sales["unit_price"] > 0]
    if paid.empty:
        return np.nan
    med = paid.groupby("item_id")["unit_price"].median()
    lp = items.set_index("item_id")["list_price"]
    j = pd.concat([med.rename("realised"), lp.rename("list")], axis=1).dropna()
    j = j[j["list"] > 0]
    if j.empty:
        return np.nan
    ratio = (j["realised"] / j["list"]).replace([np.inf, -np.inf], np.nan).dropna()
    # Trim the tails - a handful of items have list prices on a different basis.
    ratio = ratio[(ratio > 0.1) & (ratio < 3.0)]
    return float(ratio.median()) if len(ratio) else np.nan


def resolve_unit_prices(
    sales: pd.DataFrame, items: pd.DataFrame, cfg: dict,
    item_ids: pd.Index | None = None,
) -> pd.DataFrame:
    """One row per product: the unit price to use for FY forecast revenue."""
    pc = cfg["pricing"]
    window = pc.get("realised_price_window_months", 12)

    ids = pd.Index(item_ids if item_ids is not None else sales["item_id"].unique())
    ids = ids.dropna().unique()

    paid = sales[sales["unit_price"] > 0]

    # 1 - realised price in the recent window.
    # QUANTITY-WEIGHTED, not the median of the line prices. Revenue is earned
    # per unit, so the price that matters is total value divided by total units.
    # Using the median of line prices over-stated forecast revenue by ~20%,
    # because bulk lines sold at lower prices carry most of the volume but
    # count once each in a median.
    def _weighted_price(frame: pd.DataFrame) -> pd.Series:
        g = frame.assign(_val=frame["unit_price"] * frame["quantity"]).groupby("item_id")
        val, qty = g["_val"].sum(), g["quantity"].sum()
        return (val / qty.replace(0, np.nan)).dropna()

    if not paid.empty:
        last_month = sales["month"].max()
        cutoff = last_month - (window - 1)
        recent = paid[paid["month"] >= cutoff]
        recent_med = _weighted_price(recent)
        recent_n = recent.groupby("item_id")["unit_price"].size()
        all_med = _weighted_price(paid)
        all_n = paid.groupby("item_id")["unit_price"].size()
    else:
        recent_med = recent_n = all_med = all_n = pd.Series(dtype=float)

    lp = (items.set_index("item_id")["list_price"]
          if not items.empty and "list_price" in items.columns
          else pd.Series(dtype=float))
    ratio = realised_to_list_ratio(sales, items)
    if not np.isfinite(ratio):
        ratio = pc.get("fallback_realised_to_list_ratio", 0.91)
        ratio_source = "configured fallback"
    else:
        ratio_source = "computed from this dataset"

    out = pd.DataFrame(index=pd.Index(ids, name="item_id"))
    out["price_recent"] = recent_med.reindex(out.index)
    out["price_all"] = all_med.reindex(out.index)
    out["list_price"] = lp.reindex(out.index) if len(lp) else np.nan
    out["n_priced_lines_recent"] = recent_n.reindex(out.index).fillna(0).astype(int)
    out["n_priced_lines_total"] = all_n.reindex(out.index).fillna(0).astype(int)

    unit = np.full(len(out), np.nan)
    source = np.array(["PRICE_TBD"] * len(out), dtype=object)

    r = out["price_recent"].to_numpy()
    a = out["price_all"].to_numpy()
    l = out["list_price"].to_numpy()

    m1 = np.isfinite(r) & (r > 0)
    unit[m1], source[m1] = r[m1], "realised"

    m2 = ~m1 & np.isfinite(a) & (a > 0)
    unit[m2], source[m2] = a[m2], "realised_all"

    m3 = ~(m1 | m2) & np.isfinite(l) & (l > 0)
    if np.isfinite(ratio) and ratio > 0:
        unit[m3], source[m3] = l[m3] * ratio, "list_adjusted"
    else:
        unit[m3], source[m3] = l[m3], "list_raw"

    out["unit_price"] = unit
    out["price_source"] = source
    out["price_source_plain"] = out["price_source"].map(PRICE_SOURCE_PLAIN)
    out["price_is_tbd"] = out["price_source"] == "PRICE_TBD"

    # Flag products whose transactions have ever shown price 0, so the dashboard
    # can mark them distinctly even when a price was recoverable.
    if "unit_price" in sales.columns:
        ever_zero = sales.groupby("item_id")["unit_price"].apply(lambda s: bool((s == 0).any()))
        out["has_zero_price_history"] = ever_zero.reindex(out.index).fillna(False)
    else:
        out["has_zero_price_history"] = False

    out.attrs["realised_to_list_ratio"] = ratio
    out.attrs["ratio_source"] = ratio_source
    return out.reset_index()


def backtest_revenue_method(
    sales: pd.DataFrame, prices: pd.DataFrame, rates: pd.Series,
    default_rate: float, test_fy: int,
) -> dict:
    """Check the pricing method against a year we already know the answer for.

    The quantity forecast has its own back-test. This one isolates the OTHER
    half of the revenue calculation: take the ACTUAL quantities of the most
    recent fiscal year, run them through exactly the same price and billing
    logic used for the forecast, and compare against the revenue that was
    really invoiced. Any gap is pure price/mix error, not forecast error.
    """
    hist = sales[sales["fiscal_year"] == test_fy]
    if hist.empty:
        return {}

    actual_qty = hist[hist["is_demand"]].groupby("item_id")["quantity"].sum()
    actual_rev = float(
        (hist.loc[hist["is_revenue"], "quantity"]
         * hist.loc[hist["is_revenue"], "unit_price"].where(
             hist.loc[hist["is_revenue"], "unit_price"] > 0, np.nan)).sum(skipna=True)
    )

    p = prices.set_index("item_id")
    unit = p["unit_price"].reindex(actual_qty.index)
    tbd = p["price_is_tbd"].reindex(actual_qty.index).fillna(True)
    br = actual_qty.index.map(rates).to_series(index=actual_qty.index) \
        if len(rates) else pd.Series(np.nan, index=actual_qty.index)
    br = br.fillna(default_rate).clip(0, 1)

    modelled = float((actual_qty * br * unit).where(~tbd).sum(skipna=True))
    return {
        "test_fiscal_year": int(test_fy),
        "actual_invoiced_revenue": actual_rev,
        "revenue_from_actual_qty_and_our_prices": modelled,
        "error": modelled - actual_rev,
        "error_pct": (modelled - actual_rev) / actual_rev * 100 if actual_rev else np.nan,
    }


def vip_discount_adjustment(
    sales: pd.DataFrame, customers: pd.DataFrame, cfg: dict,
) -> tuple[pd.Series, dict]:
    """Per-product price multiplier for the VIP customer discount.

    APPLIED at 0.37 since 2026-08-16 (team decision: follow the dictionary).
    Set `pricing.vip_discount: 0.0` to forecast in as-recorded terms instead.
    Background of the question, kept for the record:

    The data dictionary's Customer sheet states "VIP: 37% discount from List
    Price". But in the transaction data, VIP and End User order lines carry
    almost identical prices relative to the master list price (median 0.891 vs
    0.910), and only 0.6% of VIP lines sit near 0.63x list. So `Base Price`
    appears to be a standard/reference price that does NOT already have the VIP
    discount applied.

    This matters enormously: just 10 VIP customers account for 62% of all
    quantity and 62% of gross revenue. If their invoices really are 37% below
    Base Price, forecast revenue is overstated by roughly 23%.

    It cannot be settled from the data alone - there is no invoice-amount column
    to check against - so the pipeline always reports both the discounted and
    the undiscounted revenue, whichever setting is active.

    Returns (per-item multiplier, summary dict).
    """
    disc = float(cfg.get("pricing", {}).get("vip_discount", 0.0) or 0.0)
    ids = pd.Index(sales["item_id"].dropna().unique())

    if customers.empty or "customer_type" in customers.columns is False:
        return pd.Series(1.0, index=ids), {"applied": False, "reason": "no customer data"}
    if "customer_type" not in customers.columns:
        return pd.Series(1.0, index=ids), {"applied": False, "reason": "no customer type"}

    ct = customers.set_index("customer_id")["customer_type"]
    s = sales.assign(_vip=sales["customer_id"].map(ct).eq("VIP"))
    tot = s.groupby("item_id")["quantity"].sum()
    vip = s[s["_vip"]].groupby("item_id")["quantity"].sum()
    share = (vip.reindex(tot.index).fillna(0) / tot.replace(0, np.nan)).clip(0, 1).fillna(0)

    mult = (1.0 - disc * share).reindex(ids).fillna(1.0)
    summary = {
        "applied": disc > 0,
        "vip_discount": disc,
        "vip_customers": int((ct == "VIP").sum()),
        "vip_share_of_quantity": float(
            s.loc[s["_vip"], "quantity"].sum() / max(s["quantity"].sum(), 1)),
        "average_price_multiplier": float(mult.mean()),
    }
    return mult, summary


def billing_rate(sales: pd.DataFrame, by: str = "item_id") -> pd.Series:
    """Share of demand that historically converted into billed revenue.

    Demand is forecast on every order status, but revenue must only count what
    actually gets invoiced. This ratio bridges the two.
    """
    if "is_revenue" not in sales.columns:
        return pd.Series(dtype=float)
    g = sales.groupby(by)
    billed = g.apply(lambda d: d.loc[d["is_revenue"], "quantity"].sum(), include_groups=False)
    total = g["quantity"].sum()
    rate = (billed / total.replace(0, np.nan)).clip(0, 1)
    return rate


def apply_revenue(
    forecast_long: pd.DataFrame, prices: pd.DataFrame, rates: pd.Series,
    default_rate: float,
) -> pd.DataFrame:
    """Attach unit price, billable quantity and revenue to a long forecast frame."""
    df = forecast_long.merge(
        prices[["item_id", "unit_price", "price_source", "price_source_plain",
                "price_is_tbd", "has_zero_price_history"]],
        on="item_id", how="left",
    )
    r = df["item_id"].map(rates) if len(rates) else pd.Series(np.nan, index=df.index)
    df["billing_rate"] = r.fillna(default_rate).clip(0, 1)
    df["forecast_quantity_billable"] = df["forecast_quantity"] * df["billing_rate"]
    df["forecast_revenue"] = np.where(
        df["price_is_tbd"].fillna(True), np.nan,
        df["forecast_quantity_billable"] * df["unit_price"],
    )
    df["revenue_flag"] = np.where(
        df["price_is_tbd"].fillna(True), "PRICE_TBD",
        np.where(df["has_zero_price_history"].fillna(False),
                 "PRICE_VARIABLE_ESTIMATED", "OK"),
    )
    return df
