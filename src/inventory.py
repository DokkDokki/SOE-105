"""Turning the forecast into an inventory decision.

The business problem is not "what will demand be" - it is "how much stock should
we hold". Those are different questions, and the second one has no single right
answer: holding more stock serves more demand but ties up more capital. Surplus
and shortage are two ends of one trade-off.

So the output here is not a number but a CURVE. For a range of stocking levels,
it measures - on a fiscal year that was withheld from the model - how much of the
demand that actually materialised would have been served, and how much stock
would have been held on average to achieve it. The business then chooses where on
that curve it wants to sit, and the corresponding stock levels can be applied to
the forecast year.

Everything here is measured against real withheld outcomes. Nothing is assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Safety factors evaluated along the frontier. 1.0 means "stock exactly to the
# forecast"; 2.0 means "stock to twice the forecast".
DEFAULT_FACTORS = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)


def simulate_policy(target: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    """Order-up-to simulation over a product x month matrix.

    Each month, stock is topped up to `target`; demand is served from what is on
    hand; anything left over carries into the next month.

    Returns (service level, average units held, units of demand missed).
    """
    n, T = actual.shape
    on_hand = np.zeros(n)
    met = 0.0
    held = 0.0
    for t in range(T):
        on_hand = np.maximum(on_hand, target[:, t])
        served = np.minimum(on_hand, actual[:, t])
        met += served.sum()
        on_hand -= served
        held += on_hand.sum()
    total = actual.sum()
    return (met / total if total else 1.0), held / T, total - met


def build_frontier(
    fold_store: dict, selection: pd.DataFrame, panel: pd.DataFrame, cfg: dict,
    fiscal_month_of: dict, factors=DEFAULT_FACTORS,
) -> pd.DataFrame:
    """Service level vs stock held, measured on the most recent withheld year.

    Also evaluates the same policy driven by a naive forecast, so the business
    can see how much of the benefit comes from the forecasting work itself
    rather than simply from holding more stock.
    """
    from .features import seasonal_profile
    from .forecast import apply_safeguards

    if not fold_store or selection.empty:
        return pd.DataFrame()

    # Use the most recent fold - it is the most representative of current trading.
    store = max(fold_store.values(), key=lambda s: s["test_fy"])
    ids = panel.index.to_numpy()
    sel_map = selection.set_index("item_id")["model"].to_dict()
    h = store["actual"].shape[1]

    chosen = np.full((len(ids), h), np.nan)
    for i, iid in enumerate(ids):
        m = sel_map.get(iid)
        if m and m in store["preds"] and not np.isnan(store["preds"][m][i]).any():
            chosen[i] = store["preds"][m][i]
    valid = ~np.isnan(chosen).any(axis=1)
    if valid.sum() == 0:
        return pd.DataFrame()

    train_all = store["train_values"]
    tp = pd.DataFrame(train_all, index=ids, columns=panel.columns[: store["split"]])
    seas = seasonal_profile(tp, fiscal_month_of)
    hfm = np.array([fiscal_month_of[c] for c in tp.columns])

    forecast, _, _ = apply_safeguards(
        chosen[valid].copy(), train_all[valid], hfm, seas, cfg)
    actual = store["actual"][valid]
    train = train_all[valid]

    # Naive comparator: last year's average, held flat across the year.
    naive = np.tile(train[:, -h:].sum(axis=1, keepdims=True) / h, (1, h))

    rows = []
    for label, fc in (("This forecasting system", forecast),
                      ("Naive (last year's average)", naive)):
        for f in factors:
            sl, held, missed = simulate_policy(fc * f, actual)
            rows.append({
                "Approach": label,
                "Stock factor": f,
                "Service level %": sl * 100,
                "Average stock held (units)": held,
                "Demand missed (units)": missed,
            })
    out = pd.DataFrame(rows)
    out.attrs["test_fy"] = store["test_fy"]
    out.attrs["total_demand"] = float(actual.sum())
    return out


def recommend_stock(
    forecast_long: pd.DataFrame, factor: float, round_up: bool = True,
) -> pd.DataFrame:
    """Apply a chosen stock factor to the forecast year, per product per month."""
    df = forecast_long.copy()
    df["recommended_stock"] = df["forecast_quantity"] * factor
    if round_up:
        df["recommended_stock"] = np.ceil(df["recommended_stock"])
    return df


def factor_for_service(frontier: pd.DataFrame, target_service: float,
                       approach: str = "This forecasting system") -> dict:
    """Smallest stock factor that reaches the requested service level."""
    if frontier.empty:
        return {}
    g = frontier[frontier["Approach"] == approach].sort_values("Stock factor")
    hit = g[g["Service level %"] >= target_service]
    if hit.empty:
        row = g.iloc[-1]
        return {"factor": float(row["Stock factor"]),
                "service": float(row["Service level %"]),
                "stock": float(row["Average stock held (units)"]),
                "achievable": False}
    row = hit.iloc[0]
    return {"factor": float(row["Stock factor"]),
            "service": float(row["Service level %"]),
            "stock": float(row["Average stock held (units)"]),
            "achievable": True}
