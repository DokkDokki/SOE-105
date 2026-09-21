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

# Event-buffer sizes evaluated along the frontier, as a multiple of each
# product's own P90 single-month demand. 0 means "no buffer".
#
# WHY A BUFFER AT ALL. A multiplier alone cannot reach a high service level on
# this catalogue: measured on the withheld year, 90% needs 15x the forecast and
# ~187,000 units held, because lumpy demand is not under-forecast, it is
# differently SHAPED - a product expected at 0.3 a month that occasionally takes
# one order of 50 needs a 150x multiplier, and a multiplier spends that on every
# other product too. Holding the larger of (forecast x factor) and (a multiple
# of the product's own P90 month) reaches the same 90% with ~74,000 units.
DEFAULT_BUFFERS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
EXPOSURE_QUANTILE = 0.9


def event_exposure(values: np.ndarray, q: float = EXPOSURE_QUANTILE) -> np.ndarray:
    """Per product: the size a single month can plausibly reach - the q-quantile
    of its own non-zero months (its maximum where history is too short for a
    quantile to mean anything). Spike TIMING is unpredictable (docs/IDEAS.md
    A1-A3); spike SIZE is on the record.
    """
    out = np.zeros(values.shape[0])
    for i, r in enumerate(np.asarray(values, dtype=float)):
        nz = r[r > 0]
        if len(nz):
            out[i] = np.quantile(nz, q) if len(nz) >= 4 else nz.max()
    return out


def to_whole_units(values: np.ndarray, totals: np.ndarray | None = None) -> np.ndarray:
    """Turn fractional monthly quantities into whole units, row by row,
    WITHOUT changing each row's total (largest-remainder allocation).

    Plain rounding is not an option here: a product expected to sell 0.33 a
    month rounds to zero in every month and vanishes - that once deleted 1,922
    products and 12.5% of volume (see config.yaml `output.round_quantities`).
    Instead each row is floored, and the units still owed to reach the rounded
    row total go to the months with the largest fractional parts - which, after
    the seasonal reshape, are the seasonally strongest months. Ties go to the
    EARLIER month: stock held early carries forward, stock held late cannot
    serve demand that already came.

    `totals` lets the caller fix each row's whole-unit total (the published
    annual figure); otherwise it is the rounded row sum.
    """
    v = np.clip(np.asarray(values, dtype=float), 0.0, None)
    base = np.floor(v)
    want = (np.rint(np.round(v.sum(axis=1), 6)) if totals is None
            else np.rint(np.asarray(totals, dtype=float)))
    # Never owe more units than there are months with a fractional part to
    # round up, and never fewer than zero - keeps every month within one unit.
    owed = np.clip(want - base.sum(axis=1), 0, (v > base).sum(axis=1)).astype(int)
    frac = v - base
    order = np.argsort(-frac, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(v.shape[0])[:, None]
    rank[rows, order] = np.arange(v.shape[1])[None, :]
    return base + (rank < owed[:, None])


def stock_targets(forecast: np.ndarray, factor: float,
                  exposure: np.ndarray | None = None, buffer: float = 0.0) -> np.ndarray:
    """Whole-unit order-up-to levels: the LARGER of the forecast times the
    factor and `buffer` times the product's event exposure, rounded UP.

    Stock is indivisible, so levels are whole units - and the frontier is
    simulated on these same levels, so the service level shown is the one the
    exported plan achieves. The buffer applies only to products with a live
    forecast: a product forecast at zero is one the business is not stocking.
    """
    fc = np.clip(np.asarray(forecast, dtype=float), 0.0, None)
    tgt = np.ceil(fc * factor - 1e-9)
    if exposure is not None and buffer > 0:
        live = fc.sum(axis=1) > 0
        floor = np.ceil(np.asarray(exposure, dtype=float) * buffer - 1e-9) * live
        tgt = np.maximum(tgt, floor[:, None])
    return tgt


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
    fiscal_month_of: dict, factors=DEFAULT_FACTORS, buffers=DEFAULT_BUFFERS,
) -> pd.DataFrame:
    """Service level vs stock held, measured on the most recent withheld year.

    Each point is a whole-unit stocking policy (`stock_targets`): a forecast
    multiplier plus an optional event buffer. Only the efficient policies are
    kept, so the curve reads as "the least stock that buys this service level".

    Also evaluates the same policy driven by a naive forecast, so the business
    can see how much of the benefit comes from the forecasting work itself
    rather than simply from holding more stock.
    """
    from .features import seasonal_profile
    from .forecast import apply_safeguards

    if not fold_store or selection.empty:
        return pd.DataFrame()

    # Use the most recent fold - it is the most representative of current trading.
    from .backtest import delivered_matrix

    store = max(fold_store.values(), key=lambda s: s["test_fy"])
    ids = panel.index.to_numpy()
    h = store["actual"].shape[1]

    chosen = delivered_matrix(store["preds"], selection, ids, h, cfg)
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

    # Event exposure from the TRAINING window only - the test year must not
    # inform the buffer it is scored against.
    exposure = event_exposure(train)

    rows = []
    for label, fc in (("This forecasting system", forecast),
                      ("Naive (last year's average)", naive)):
        tried = []
        for f in factors:
            for b in buffers:
                sl, held, missed = simulate_policy(
                    stock_targets(fc, f, exposure, b), actual)
                tried.append((held, sl * 100, f, b, missed))
        # Keep the efficient ladder only: a policy stays if it serves more
        # demand than every cheaper one. Anything else is simply a worse buy.
        tried.sort()
        best = -1.0
        for held, sl, f, b, missed in tried:
            if sl > best + 0.05:
                best = sl
                rows.append({
                    "Approach": label,
                    "Stock factor": f,
                    "Event buffer (x P90 month)": b,
                    "Service level %": sl,
                    "Average stock held (units)": held,
                    "Demand missed (units)": missed,
                })
    out = pd.DataFrame(rows)
    out.attrs["test_fy"] = store["test_fy"]
    out.attrs["total_demand"] = float(actual.sum())
    return out


def recommend_stock(forecast_long: pd.DataFrame, factor: float,
                    annual_totals: pd.Series | None = None,
                    exposure: pd.Series | None = None,
                    buffer: float = 0.0) -> pd.DataFrame:
    """The stock plan for the forecast year, per product per month, in WHOLE units.

    `forecast_quantity` becomes whole units by `to_whole_units` (each product's
    annual total is preserved; pass `annual_totals` - item_id -> published
    full-year figure - and the file adds up to the published headline exactly).
    `recommended_stock` is the whole-unit order-up-to level from `stock_targets`
    (`exposure`: item_id -> P90 single-month demand; `buffer`: its multiple),
    computed from the underlying expected demand - the identical rule the
    frontier was simulated with, so the service level quoted on the page is the
    one this file delivers.
    """
    wide = forecast_long.pivot_table(
        index="item_id", columns="month", values="forecast_quantity",
        aggfunc="sum", sort=True).fillna(0.0)
    months = list(wide.columns)
    expected = wide.to_numpy(dtype=float)

    def _long(arr: np.ndarray, name: str) -> pd.DataFrame:
        return (pd.DataFrame(arr, index=wide.index,
                             columns=pd.Index(months, name="month"))
                .stack().rename(name).reset_index())

    expo = (None if exposure is None else
            exposure.reindex(wide.index).fillna(0.0).to_numpy(dtype=float))
    totals = (None if annual_totals is None
              else annual_totals.reindex(wide.index).to_numpy(dtype=float))
    if totals is not None:
        totals = np.where(np.isnan(totals), expected.sum(axis=1), totals)
    out = _long(to_whole_units(expected, totals), "forecast_quantity").merge(
        _long(stock_targets(expected, factor, expo, buffer), "recommended_stock"),
        on=["item_id", "month"])
    for c in ("forecast_quantity", "recommended_stock"):
        out[c] = out[c].round().astype("int64")

    keep = [c for c in forecast_long.columns
            if c not in ("month", "forecast_quantity")]
    if len(keep) > 1:
        out = out.merge(forecast_long[keep].drop_duplicates("item_id"),
                        on="item_id", how="left")
        out = out[keep + ["month", "forecast_quantity", "recommended_stock"]]
    return out


def factor_for_service(frontier: pd.DataFrame, target_service: float,
                       approach: str = "This forecasting system") -> dict:
    """Cheapest stocking policy that reaches the requested service level."""
    if frontier.empty:
        return {}
    g = (frontier[frontier["Approach"] == approach]
         .sort_values("Average stock held (units)"))
    hit = g[g["Service level %"] >= target_service]
    row = g.iloc[-1] if hit.empty else hit.iloc[0]
    return {"factor": float(row["Stock factor"]),
            # frontiers saved before the event buffer existed have no such column
            "buffer": float(row.get("Event buffer (x P90 month)", 0.0) or 0.0),
            "service": float(row["Service level %"]),
            "stock": float(row["Average stock held (units)"]),
            "achievable": not hit.empty}
