"""Refit the selected models on all available history and produce the forecast,
then optionally reconcile product forecasts to independently-forecast category
totals.

Whether reconciliation helps is decided by the back-test, not assumed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import models as MOD
from .backtest import _run_model_on_products
from .features import TIER_DORMANT, seasonal_profile
from .global_ml import GlobalMLForecaster

warnings.filterwarnings("ignore")


def _forecast_with_registry(
    values: np.ndarray, model_names: np.ndarray, registry: dict, cfg: dict,
    hist_fms: np.ndarray, future_fms: list[int], h: int, seas_prof: np.ndarray,
    n_jobs: int, panel: pd.DataFrame, items: pd.DataFrame, fiscal_month_of: dict,
    progress=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run each product's chosen model. Returns (forecast, fallback_used flags)."""
    n, out = values.shape[0], np.full((values.shape[0], h), np.nan)
    ctx_base = {
        "halflife": cfg["models"].get("recency_halflife_months", 18),
        "croston_alpha": cfg["models"].get("croston_alpha", 0.1),
        "seasonal_profile": seas_prof,
    }

    needed = pd.unique(model_names[pd.notna(model_names)])

    # The pooled ML model predicts every product in one pass.
    if "global_ml" in needed and cfg["models"].get("enable_global_ml", True):
        try:
            gm = GlobalMLForecaster(horizon=h).fit(panel, items, fiscal_month_of)
            pred = gm.predict(panel, items, fiscal_month_of, future_fms).to_numpy()
            rows = np.flatnonzero(model_names == "global_ml")
            out[rows] = pred[rows]
        except Exception:
            pass
        if progress:
            progress("global_ml")

    for m in needed:
        if m in ("global_ml", "zero"):
            continue
        fn = registry.get(m)
        rows = np.flatnonzero(model_names == m)
        if fn is None or rows.size == 0:
            continue
        if rows.size < 400 or m in MOD.BASELINE_MODELS or m in MOD.INTERMITTENT_MODELS:
            ok, preds = _run_model_on_products(
                fn, values, rows, h, ctx_base, hist_fms, future_fms)
        else:
            chunks = np.array_split(rows, max(n_jobs if n_jobs > 0 else 8, 1) * 2)
            res = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run_model_on_products)(
                    fn, values, c, h, ctx_base, hist_fms, future_fms)
                for c in chunks if len(c))
            ok_p = [r[0] for r in res if len(r[0])]
            pr_p = [r[1] for r in res if len(r[0])]
            ok = np.concatenate(ok_p) if ok_p else np.array([], dtype=int)
            preds = np.vstack(pr_p) if pr_p else np.zeros((0, h))
        if len(ok):
            out[ok] = preds
        if progress:
            progress(m)

    out[model_names == "zero"] = 0.0

    # Safety net: if a chosen model failed at refit time, fall back to a
    # 6-month average rather than dropping the product from the forecast.
    failed = np.isnan(out).any(axis=1)
    if failed.any():
        rows = np.flatnonzero(failed)
        fb_rows, fb_preds = _run_model_on_products(
            MOD.moving_average_6, values, rows, h, ctx_base, hist_fms, future_fms)
        if len(fb_rows):
            out[fb_rows] = fb_preds
    out = np.nan_to_num(out, nan=0.0)
    return np.clip(out, 0, None), failed


def forecast_group_totals(
    group_panel: pd.DataFrame, cfg: dict, fy_of: dict, fiscal_month_of: dict,
    future_fms: list[int], h: int,
) -> pd.DataFrame:
    """Forecast each product-category total independently.

    Category series are dense and well behaved even when their member products
    are not, so they carry much more reliable trend and seasonality information.
    """
    if group_panel.empty:
        return pd.DataFrame()

    registry = MOD.build_registry(cfg)
    candidates = [m for m in registry if m not in ("zero",)]
    values = group_panel.to_numpy(dtype=float)
    months = list(group_panel.columns)
    all_fms = np.array([fiscal_month_of[m] for m in months])
    fys = np.array([fy_of[m] for m in months])

    # Hold out the last fiscal year to pick each category's model.
    last_fy = fys.max()
    test_mask = fys == last_fy
    split = int((~test_mask).sum())
    if split < 24 or test_mask.sum() == 0:
        split = max(len(months) - h, 12)
        test_mask = np.zeros(len(months), bool)
        test_mask[split:] = True

    seas_prof = seasonal_profile(group_panel.iloc[:, :split], fiscal_month_of)
    ctx = {
        "halflife": cfg["models"].get("recency_halflife_months", 18),
        "croston_alpha": cfg["models"].get("croston_alpha", 0.1),
        "seasonal_profile": seas_prof,
        "hist_fiscal_months": all_fms[:split],
        "future_fiscal_months": [int(x) for x in all_fms[test_mask]],
    }
    actual = values[:, test_mask]
    th = actual.shape[1]

    bias_weight = cfg["backtest"].get("bias_weight", 1.0)
    best = {}
    for gi in range(values.shape[0]):
        y_tr = values[gi, :split]
        scores = {}
        for m in candidates:
            try:
                p = registry[m](y_tr, th, ctx)
            except Exception:
                p = None
            if p is None or len(p) != th or not np.isfinite(p).all():
                continue
            a = actual[gi]
            denom = np.abs(a).sum()
            if denom <= 0:
                continue
            # Same combined criterion as product-level selection: per-month
            # accuracy plus a penalty for getting the annual total wrong.
            wape = np.abs(a - p).sum() / denom
            rel_bias = abs(p.sum() - a.sum()) / denom
            scores[m] = wape + bias_weight * rel_bias
        best[gi] = min(scores, key=scores.get) if scores else "moving_average_6"

    # Refit the winner on the full series.
    ctx_full = dict(ctx)
    ctx_full["hist_fiscal_months"] = all_fms
    ctx_full["future_fiscal_months"] = future_fms
    ctx_full["seasonal_profile"] = seasonal_profile(group_panel, fiscal_month_of)

    rows = []
    for gi, name in best.items():
        try:
            p = registry[name](values[gi], h, ctx_full)
        except Exception:
            p = None
        if p is None or len(p) != h or not np.isfinite(p).all():
            p = MOD.moving_average_6(values[gi], h, ctx_full)
            name = "moving_average_6"
        rows.append({"group": group_panel.index[gi], "model": name,
                     **{f"h{j+1}": float(v) for j, v in enumerate(p)}})
    out = pd.DataFrame(rows).set_index("group")
    return out


# Forecast columns are named h1..hN; "model" must never be picked up as one.
def _horizon_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if isinstance(c, str) and c.startswith("h")
            and c[1:].isdigit()]


def apply_safeguards(
    fc: np.ndarray, values: np.ndarray, hist_fms: np.ndarray,
    seas_prof: np.ndarray, cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two business safeguards applied to every product forecast.

    FLOOR (stock-shortage guard)
        A product that demonstrably sold in the last 12 months is never
        forecast at zero for a whole year. The back-test cannot catch this: for
        a lumpy product whose test year happened to contain no sales, a zero
        forecast scores perfectly, so zero wins on the evidence alone. Planning
        zero for a product that does sell guarantees a stockout. The floor is
        LAST YEAR'S OWN VOLUME - not a lifetime average, which would ignore a
        product in genuine decline.

    CAP (stock-surplus guard)
        No product is forecast above a modest multiple of the largest annual
        volume it has achieved in the last three fiscal years. Simple methods
        such as "repeat last month" or a straight-line trend can take one
        unusually large final month and project it across twelve, which is how
        a forecast quietly turns into surplus stock. There is no evidence for
        volumes beyond historical peaks, so the forecast is scaled back to one.

    Returns (adjusted forecast, floor-applied flags, cap-applied flags).
    """
    sg = cfg.get("safeguards", {})
    n, h = fc.shape
    floor_used = np.zeros(n, dtype=bool)
    cap_used = np.zeros(n, dtype=bool)

    last12 = values[:, -12:] if values.shape[1] >= 12 else values
    last12_total = last12.sum(axis=1)
    annual_fc = fc.sum(axis=1)

    # ------------------------------------------------------------- floor
    if sg.get("never_zero_if_sold_recently", True):
        rows = np.flatnonzero((last12_total > 0) & (annual_fc <= 0))
        if rows.size:
            # Spread last year's volume across the year using the shared
            # seasonal shape rather than a flat line.
            shape = np.asarray(seas_prof, dtype=float)
            shape = shape / shape.sum() if shape.sum() > 0 else np.full(12, 1 / 12)
            spread = np.tile(shape[:h] / shape[:h].sum(), (rows.size, 1))
            fc[rows] = spread * last12_total[rows, None]
            floor_used[rows] = True
            annual_fc = fc.sum(axis=1)

    # --------------------------------------------------------------- cap
    growth_cap = sg.get("max_growth_vs_recent_peak", 1.5)
    if growth_cap and growth_cap > 0 and values.shape[1] >= 12:
        # Annual volumes for the last three complete years of history.
        n_years = min(3, values.shape[1] // 12)
        peaks = np.zeros(n)
        for k in range(n_years):
            block = values[:, values.shape[1] - 12 * (k + 1): values.shape[1] - 12 * k]
            peaks = np.maximum(peaks, block.sum(axis=1))
        cap = peaks * growth_cap
        rows = np.flatnonzero((cap > 0) & (annual_fc > cap))
        if rows.size:
            scale = (cap[rows] / annual_fc[rows])[:, None]
            fc[rows] = fc[rows] * scale
            cap_used[rows] = True

    return np.clip(fc, 0, None), floor_used, cap_used


def reconcile_to_groups(
    item_fc: np.ndarray, item_ids: np.ndarray, group_of: pd.Series,
    group_fc: pd.DataFrame, max_adjustment: float = 1.5,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Scale product forecasts so they sum to the category forecast.

    Adjustment is capped so a single odd category total cannot distort every
    product inside it.
    """
    if group_fc.empty:
        return item_fc, pd.DataFrame()

    h = item_fc.shape[1]
    groups = pd.Series(item_ids).map(group_of).fillna("Unknown").to_numpy()
    out = item_fc.copy()
    audit = []

    gcols = _horizon_cols(group_fc)
    for g in pd.unique(groups):
        if g not in group_fc.index:
            continue
        rows = np.flatnonzero(groups == g)
        bottom_up = item_fc[rows].sum(axis=0)
        target = group_fc.loc[g, gcols].to_numpy(dtype=float)[:h]
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(bottom_up > 0, target / bottom_up, 1.0)
        factor = np.clip(np.nan_to_num(factor, nan=1.0), 1 / max_adjustment, max_adjustment)
        out[rows] = item_fc[rows] * factor
        audit.append({
            "Category": g, "Products": len(rows),
            "Bottom-up total": float(bottom_up.sum()),
            "Category forecast": float(target.sum()),
            "Average adjustment": float(np.mean(factor)),
        })
    return out, pd.DataFrame(audit)


def test_reconciliation(
    fold_store: dict, selection: pd.DataFrame, panel: pd.DataFrame,
    items: pd.DataFrame, cfg: dict, fy_of: dict, fiscal_month_of: dict,
) -> dict:
    """Does reconciliation actually improve accuracy? Measure it on the folds.

    The answer is written to outputs/ so the decision is auditable rather than
    a matter of taste.
    """
    from .features import build_group_panel  # noqa: F401  (kept for clarity)

    level = cfg["reconciliation"].get("level", "category_1")
    level_col = {"Product Category 1": "category_1", "Product Category 2": "category_2"}.get(
        level, level)
    if items.empty or level_col not in items.columns:
        return {"enabled": False, "reason": "No product category available to reconcile to."}

    group_of = items.set_index("item_id")[level_col]
    sel_map = selection.set_index("item_id")["model"].to_dict()
    item_ids = panel.index.to_numpy()
    verdicts = []

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
        cv, av, ids = chosen[valid], actual[valid], item_ids[valid]

        # Category totals from the training window only.
        train_vals = store["train_values"]
        train_panel = pd.DataFrame(train_vals, index=item_ids,
                                   columns=panel.columns[: store["split"]])
        gmap = pd.Series(item_ids).map(group_of).fillna("Unknown")
        gp = train_panel.groupby(gmap.to_numpy()).sum()
        future_fms = [int(fiscal_month_of[panel.columns[i]]) for i in store["test_idx"]]
        gfc = forecast_group_totals(gp, cfg, fy_of, fiscal_month_of, future_fms, h)

        rec, _ = reconcile_to_groups(cv, ids, group_of, gfc,
                                     cfg["reconciliation"].get("max_adjustment", 2.0))
        denom = np.abs(av).sum()
        if denom <= 0:
            continue
        bw = cfg["backtest"].get("bias_weight", 1.0)

        def _sc(pred):
            wape = float(np.abs(av - pred).sum() / denom)
            tot_err = float(abs(pred.sum() - av.sum()) / denom)
            return wape, tot_err, wape + bw * tot_err

        w_bu, t_bu, c_bu = _sc(cv)
        w_rc, t_rc, c_rc = _sc(rec)
        verdicts.append({
            "fold": fold_name,
            "wape_bottom_up": w_bu, "wape_reconciled": w_rc,
            "total_error_bottom_up_pct": t_bu * 100, "total_error_reconciled_pct": t_rc * 100,
            "combined_bottom_up": c_bu, "combined_reconciled": c_rc,
        })

    if not verdicts:
        return {"enabled": False, "reason": "Could not be tested.", "folds": []}

    judge = cfg["reconciliation"].get("judge_on", "combined")
    key = {"combined": "combined", "wape": "wape", "total_error": "total_error"}.get(
        judge, "combined")
    if key == "combined":
        bu = float(np.mean([v["combined_bottom_up"] for v in verdicts]))
        rc = float(np.mean([v["combined_reconciled"] for v in verdicts]))
        metric_name = "combined score (accuracy + total-level bias)"
    elif key == "wape":
        bu = float(np.mean([v["wape_bottom_up"] for v in verdicts]))
        rc = float(np.mean([v["wape_reconciled"] for v in verdicts]))
        metric_name = "WAPE"
    else:
        bu = float(np.mean([v["total_error_bottom_up_pct"] for v in verdicts]))
        rc = float(np.mean([v["total_error_reconciled_pct"] for v in verdicts]))
        metric_name = "total-error %"

    enabled = bool(rc < bu) and cfg["reconciliation"].get("enabled", True)
    return {
        "enabled": enabled,
        "judged_on": metric_name,
        "score_bottom_up": bu,
        "score_reconciled": rc,
        "wape_bottom_up": float(np.mean([v["wape_bottom_up"] for v in verdicts])),
        "wape_reconciled": float(np.mean([v["wape_reconciled"] for v in verdicts])),
        "total_error_bottom_up_pct": float(
            np.mean([v["total_error_bottom_up_pct"] for v in verdicts])),
        "total_error_reconciled_pct": float(
            np.mean([v["total_error_reconciled_pct"] for v in verdicts])),
        "folds": verdicts,
        "reason": (
            f"Scaling product forecasts to independently-forecast {level} totals "
            f"improved the {metric_name} from {bu:.3f} to {rc:.3f} in back-testing, "
            "so it is applied."
            if enabled else
            f"Scaling product forecasts to {level} totals did not improve the "
            f"{metric_name} ({bu:.3f} as-is vs {rc:.3f} scaled), so product "
            "forecasts are used exactly as the per-product models produced them."
        ),
    }


def build_forecast(
    panel: pd.DataFrame, items: pd.DataFrame, selection: pd.DataFrame, cfg: dict,
    fy_of: dict, fiscal_month_of: dict, future_index: pd.PeriodIndex,
    reconciliation_decision: dict | None = None, n_jobs: int = -1, progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce the final forecast. Returns (long forecast, group forecast, audit)."""
    registry = MOD.build_registry(cfg)
    h = len(future_index)
    values = panel.to_numpy(dtype=float)
    months = list(panel.columns)
    all_fms = np.array([fiscal_month_of[m] for m in months])
    future_fms = [int(fiscal_month_of[m]) for m in future_index]

    sel = selection.set_index("item_id").reindex(panel.index)
    model_names = sel["model"].fillna("moving_average_6").to_numpy()

    seas_prof = seasonal_profile(panel, fiscal_month_of)
    fc, failed = _forecast_with_registry(
        values, model_names, registry, cfg, all_fms, future_fms, h, seas_prof,
        n_jobs, panel, items, fiscal_month_of, progress,
    )

    # ------------------------------------------------------------ safeguards
    fc, guard_used, cap_used = apply_safeguards(fc, values, all_fms, seas_prof, cfg)

    # -------------------------------------------------------- reconciliation
    level_col = {"Product Category 1": "category_1",
                 "Product Category 2": "category_2"}.get(
        cfg["reconciliation"].get("level", "category_1"), "category_1")
    group_fc, audit = pd.DataFrame(), pd.DataFrame()
    if (not items.empty and level_col in items.columns
            and (reconciliation_decision or {}).get("enabled", False)):
        group_of = items.set_index("item_id")[level_col]
        gmap = pd.Series(panel.index).map(group_of).fillna("Unknown")
        gp = panel.groupby(gmap.to_numpy()).sum()
        group_fc = forecast_group_totals(gp, cfg, fy_of, fiscal_month_of, future_fms, h)
        fc, audit = reconcile_to_groups(
            fc, panel.index.to_numpy(), group_of, group_fc,
            cfg["reconciliation"].get("max_adjustment", 2.0))

    if cfg["output"].get("round_quantities", False):
        fc = np.round(fc, 0)
    else:
        # Keep expected demand fractional so slow movers survive; two decimals
        # is plenty for planning and keeps the CSV readable.
        fc = np.round(fc, 2)

    long = pd.DataFrame(fc, index=panel.index, columns=future_index)
    long.index.name = "item_id"
    long = long.stack().rename("forecast_quantity").reset_index()
    long.columns = ["item_id", "month", "forecast_quantity"]
    long["fiscal_month"] = long["month"].map(fiscal_month_of)
    long = long.merge(
        selection[["item_id", "model", "model_plain", "tier", "backtest_WAPE",
                   "backtest_MASE", "backtest_MAE", "backtest_MAPE",
                   "selection_reason"]],
        on="item_id", how="left")
    long["refit_failed"] = long["item_id"].map(
        dict(zip(panel.index, failed))).fillna(False)
    long["shortage_floor_applied"] = long["item_id"].map(
        dict(zip(panel.index, guard_used))).fillna(False)
    long["surplus_cap_applied"] = long["item_id"].map(
        dict(zip(panel.index, cap_used))).fillna(False)

    # Keep the displayed method honest when a safeguard changed the answer.
    long.loc[long["shortage_floor_applied"], "model_plain"] = (
        "Last year's own volume (stock-shortage safeguard)")
    long.loc[long["surplus_cap_applied"], "model_plain"] = (
        long.loc[long["surplus_cap_applied"], "model_plain"].astype(str)
        + " - capped at recent peak")
    return long, group_fc, audit
