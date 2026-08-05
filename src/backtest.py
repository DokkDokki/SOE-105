"""Back-testing and per-product model selection.

There are no FY2025 actuals to check against, so accuracy is established by
hiding the last year(s) of the data we DO have and forecasting them.

Two folds are run:
  Fold 1 - train FY2021-FY2022, forecast FY2023
  Fold 2 - train FY2021-FY2023, forecast FY2024   (the fold the brief asked for)

Fold 2 alone would be risky: its test year sits immediately after the sharp
FY2022->FY2023 drop, so it rewards whichever model happened to catch the
rebound. Requiring a model to do well on both folds guards against that.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import metrics as M
from . import models as MOD
from .features import (TIER_DENSE, TIER_DORMANT, TIER_MODERATE, TIER_SPARSE,
                       seasonal_profile)
from .global_ml import GlobalMLForecaster

warnings.filterwarnings("ignore")

# Models that are cheap enough to run on every product without parallel help.
_CHEAP = set(MOD.BASELINE_MODELS) | set(MOD.INTERMITTENT_MODELS) | set(MOD.PROFILE_MODELS)


@dataclass
class Fold:
    name: str
    train_end_fy: int
    test_fy: int
    split_idx: int
    test_idx: np.ndarray


def _tier_from_train(active_months: np.ndarray, sold_recently: np.ndarray, cfg: dict) -> np.ndarray:
    """Assign tiers using ONLY the training window, so nothing leaks."""
    seg = cfg["segmentation"]
    return np.where(
        ~sold_recently, TIER_DORMANT,
        np.where(active_months >= seg["dense_min_active_months"], TIER_DENSE,
                 np.where(active_months >= seg["moderate_min_active_months"],
                          TIER_MODERATE, TIER_SPARSE)),
    )


def auto_fold_specs(fys: np.ndarray, cfg: dict) -> list[dict]:
    """Generate fold definitions from whatever history is actually present.

    This is what makes the project survive the passage of time. With the folds
    written out by hand, adding a new year of data would leave the pipeline
    still testing itself against the same two old years - quietly going stale
    while looking like it worked. Generated folds always test the most recent
    complete fiscal years available.

    It also fixes itself: the seasonal models (Holt-Winters, SARIMA) need 25-28
    months of training data and today's earliest fold only has 24, so they are
    excluded (see docs/ASSUMPTIONS.md section 16). Once one more year of
    history exists, every fold clears that bar and the seasonal models start
    competing automatically, with no code change.
    """
    bt = cfg["backtest"]
    n_folds = bt.get("auto_n_folds", 2)
    min_train_months = bt.get("auto_min_train_months", 24)

    counts = pd.Series(fys).value_counts().sort_index()
    complete = [int(fy) for fy, n in counts.items() if n >= 12]
    if len(complete) < 2:
        return []

    specs = []
    for test_fy in complete[-n_folds:]:
        train_months = int((fys < test_fy).sum())
        if train_months < min_train_months:
            continue
        specs.append({"name": f"Test FY{test_fy}", "train_end_fy": test_fy - 1,
                      "test_fy": test_fy})
    return specs


def _auto_weights(n: int) -> list[float]:
    """Recent folds matter more: weights rise linearly, then normalise."""
    if n <= 0:
        return []
    raw = [i + 1 for i in range(n)]
    total = sum(raw)
    return [r / total for r in raw]


def build_folds(panel: pd.DataFrame, cfg: dict, fy_of: dict) -> list[Fold]:
    """Turn the configured (or generated) fold definitions into index ranges."""
    months = list(panel.columns)
    fys = np.array([fy_of[m] for m in months])

    spec_cfg = cfg["backtest"].get("folds")
    if spec_cfg in (None, "auto", "AUTO"):
        specs = auto_fold_specs(fys, cfg)
        cfg["backtest"]["_resolved_folds"] = specs
        if not cfg["backtest"].get("fold_weights"):
            cfg["backtest"]["fold_weights"] = _auto_weights(len(specs))
    else:
        specs = spec_cfg
        cfg["backtest"]["_resolved_folds"] = specs

    folds: list[Fold] = []
    for spec in specs:
        test_mask = fys == spec["test_fy"]
        train_mask = fys <= spec["train_end_fy"]
        if test_mask.sum() == 0 or train_mask.sum() < 12:
            continue
        split = int(train_mask.sum())
        test_idx = np.flatnonzero(test_mask)
        if test_idx.min() < split:
            continue
        folds.append(Fold(spec["name"], spec["train_end_fy"], spec["test_fy"],
                          split, test_idx))
    return folds


def _run_model_on_products(
    model_fn, values: np.ndarray, rows: np.ndarray, h: int, ctx_base: dict,
    hist_fms: np.ndarray, future_fms: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Run one model over a subset of products. Returns (row_ids, predictions)."""
    preds, ok_rows = [], []
    ctx = dict(ctx_base)
    ctx["hist_fiscal_months"] = hist_fms
    ctx["future_fiscal_months"] = future_fms
    for r in rows:
        y = values[r]
        try:
            out = model_fn(y, h, ctx)
        except Exception:
            out = None
        if out is None:
            continue
        out = np.asarray(out, dtype=float)
        if out.shape[0] != h or not np.isfinite(out).all():
            continue
        preds.append(out)
        ok_rows.append(r)
    if not ok_rows:
        return np.array([], dtype=int), np.zeros((0, h))
    return np.array(ok_rows, dtype=int), np.vstack(preds)


def _predict_all_models(
    train_values: np.ndarray, tiers: np.ndarray, registry: dict, cfg: dict,
    hist_fms: np.ndarray, future_fms: list[int], h: int, seas_prof: np.ndarray,
    n_jobs: int, panel_index, train_panel: pd.DataFrame | None = None,
    items: pd.DataFrame | None = None, fiscal_month_of: dict | None = None,
    progress=None,
) -> dict[str, np.ndarray]:
    """Predictions from every model for every eligible product.

    Result maps model name -> (n_products, h) array, NaN where the model was
    not applicable to that product.
    """
    n = train_values.shape[0]
    ctx_base = {
        "halflife": cfg["models"].get("recency_halflife_months", 18),
        "croston_alpha": cfg["models"].get("croston_alpha", 0.1),
        "seasonal_profile": seas_prof,
    }

    # Which products may use which model, from the training-window tier only.
    eligible: dict[str, np.ndarray] = {}
    for tier in (TIER_DENSE, TIER_MODERATE, TIER_SPARSE):
        rows = np.flatnonzero(tiers == tier)
        if rows.size == 0:
            continue
        for m in MOD.models_for_tier(tier, registry):
            eligible.setdefault(m, []).append(rows)
    eligible = {m: np.concatenate(v) for m, v in eligible.items()}

    out: dict[str, np.ndarray] = {}

    for model_name, rows in eligible.items():
        if rows.size == 0:
            continue
        fn = registry[model_name]
        arr = np.full((n, h), np.nan)

        if model_name in _CHEAP or rows.size < 400:
            ok, preds = _run_model_on_products(
                fn, train_values, rows, h, ctx_base, hist_fms, future_fms)
        else:
            chunks = np.array_split(rows, max(n_jobs * 2, 1))
            results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                delayed(_run_model_on_products)(
                    fn, train_values, c, h, ctx_base, hist_fms, future_fms)
                for c in chunks if len(c)
            )
            ok_parts = [r[0] for r in results if len(r[0])]
            pr_parts = [r[1] for r in results if len(r[0])]
            ok = np.concatenate(ok_parts) if ok_parts else np.array([], dtype=int)
            preds = np.vstack(pr_parts) if pr_parts else np.zeros((0, h))

        if len(ok):
            arr[ok] = preds
        out[model_name] = arr
        if progress:
            progress(model_name)

    # ------------------------------------------------- pooled ML model
    if cfg["models"].get("enable_global_ml", True) and train_panel is not None:
        try:
            gm = GlobalMLForecaster(horizon=h).fit(train_panel, items, fiscal_month_of)
            pred = gm.predict(train_panel, items, fiscal_month_of, future_fms)
            arr = np.full((n, h), np.nan)
            mask = tiers != TIER_DORMANT
            arr[mask] = pred.to_numpy()[mask]
            out["global_ml"] = arr
            if progress:
                progress("global_ml")
        except Exception:
            pass

    return out


def run_backtest(
    panel: pd.DataFrame, items: pd.DataFrame, cfg: dict, fy_of: dict,
    fiscal_month_of: dict, n_jobs: int = -1, progress=None,
) -> tuple[pd.DataFrame, dict]:
    """Score every candidate model on every fold.

    Returns a long results frame (one row per product x fold x model) and a
    dict of per-fold prediction arrays for later reconciliation testing.
    """
    registry = MOD.build_registry(cfg)
    folds = build_folds(panel, cfg, fy_of)
    if not folds:
        raise ValueError(
            "Not enough history to back-test. At least two fiscal years plus a "
            "test year are required."
        )

    months = list(panel.columns)
    all_fms = np.array([fiscal_month_of[m] for m in months])
    values = panel.to_numpy(dtype=float)

    rows: list[pd.DataFrame] = []
    fold_store: dict[str, dict] = {}

    for fold in folds:
        split = fold.split_idx
        train_vals = values[:, :split]
        test_idx = fold.test_idx
        h = len(test_idx)
        actual = values[:, test_idx]

        # Training-window tiering (no leakage).
        active = (train_vals > 0).sum(axis=1)
        recent_window = train_vals[:, -12:] if train_vals.shape[1] >= 12 else train_vals
        sold_recently = recent_window.sum(axis=1) > 0
        tiers = _tier_from_train(active, sold_recently, cfg)

        train_panel = panel.iloc[:, :split]
        seas_prof = seasonal_profile(train_panel, fiscal_month_of)
        hist_fms = all_fms[:split]
        future_fms = [int(all_fms[i]) for i in test_idx]

        preds = _predict_all_models(
            train_vals, tiers, registry, cfg, hist_fms, future_fms, h, seas_prof,
            n_jobs, panel.index, train_panel, items, fiscal_month_of, progress,
        )
        # Dormant products are forecast as zero by definition.
        zero_arr = np.full((values.shape[0], h), np.nan)
        zero_arr[tiers == TIER_DORMANT] = 0.0
        preds["zero"] = zero_arr

        fold_store[fold.name] = {
            "preds": preds, "actual": actual, "tiers": tiers,
            "train_values": train_vals, "test_idx": test_idx,
            "split": split, "test_fy": fold.test_fy,
        }

        # ------------------------------------------------------- scoring
        for model_name, arr in preds.items():
            valid = ~np.isnan(arr).any(axis=1)
            if valid.sum() == 0:
                continue
            idx = np.flatnonzero(valid)
            a, p = actual[idx], arr[idx]

            abs_err = np.abs(a - p)
            denom = np.abs(a).sum(axis=1)
            wape_v = np.where(denom > 0, abs_err.sum(axis=1) / np.where(denom > 0, denom, 1),
                              np.where(np.abs(p).sum(axis=1) == 0, 0.0, np.inf))

            with np.errstate(divide="ignore", invalid="ignore"):
                nz = a != 0
                mape_num = np.where(nz, np.abs((a - p) / np.where(nz, a, 1)), np.nan)
                mape_v = np.nanmean(mape_num, axis=1) * 100.0
                skipped = (~nz).sum(axis=1)

            tr = train_vals[idx]
            m = 12 if tr.shape[1] > 12 else 1
            scale = np.abs(tr[:, m:] - tr[:, :-m]).mean(axis=1) if tr.shape[1] > m else np.zeros(len(idx))
            scale = np.where(scale > 0, scale, np.nan)
            mase_v = abs_err.mean(axis=1) / scale

            rows.append(pd.DataFrame({
                "item_id": panel.index.to_numpy()[idx],
                "fold": fold.name,
                "test_fy": fold.test_fy,
                "model": model_name,
                "tier": tiers[idx],
                "MAE": abs_err.mean(axis=1),
                "RMSE": np.sqrt(((a - p) ** 2).mean(axis=1)),
                "WAPE": wape_v,
                "MAPE": mape_v,
                "MAPE_months_skipped": skipped,
                "MASE": mase_v,
                "Bias": (p - a).mean(axis=1),
                "actual_total": a.sum(axis=1),
                "forecast_total": p.sum(axis=1),
                "n_months": h,
            }))

    results = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return results, fold_store


def select_models(results: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Choose one model per product: best weighted WAPE, with an honesty guard.

    A 12-month test window is short enough that a complicated model can win by
    luck. So a non-baseline model is only chosen if it beats the best simple
    baseline by a clear margin; otherwise the simple model wins. This trades a
    little back-test flattery for real out-of-sample robustness.
    """
    if results.empty:
        return pd.DataFrame()

    bt = cfg["backtest"]
    # `_resolved_folds` is set by build_folds and handles both hand-written and
    # generated fold definitions.
    specs = bt.get("_resolved_folds") or bt.get("folds") or []
    if isinstance(specs, str):
        specs = []
    fold_names = [f["name"] for f in specs]
    weights = bt.get("fold_weights") or [1.0] * len(fold_names)
    wmap = {n: w for n, w in zip(fold_names, weights)}
    margin = bt.get("selection_margin", 0.05)
    baselines = set(bt.get("baseline_models", []))

    bias_weight = bt.get("bias_weight", 1.0)

    df = results.copy()
    df["WAPE"] = df["WAPE"].replace([np.inf, -np.inf], np.nan)
    df["w"] = df["fold"].map(wmap).fillna(1.0)

    # Penalty for getting the ANNUAL TOTAL wrong. Without this, minimising
    # absolute error collapses intermittent products to a zero forecast,
    # because zero is the median of a series that is zero most months.
    act, fc = df["actual_total"].to_numpy(float), df["forecast_total"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_bias = np.where(act > 0, np.abs(fc - act) / act,
                            np.where(fc > 0, 1.0, 0.0))
    df["rel_bias"] = np.clip(np.nan_to_num(rel_bias, nan=1.0), 0, 5)
    df["fold_score"] = df["WAPE"] + bias_weight * df["rel_bias"]

    # Weighted mean score per product x model, over the folds it ran on.
    # The weight must only count folds where a score actually exists. Summing
    # the full weight while pandas silently skips a NaN score would make a
    # model that FAILED on one fold look better than one that worked on both.
    df["score_w"] = df["fold_score"] * df["w"]
    df["wape_w"] = df["WAPE"] * df["w"]
    df["w_valid"] = df["w"].where(df["fold_score"].notna())
    agg = df.groupby(["item_id", "model"], observed=True).agg(
        score_sum=("score_w", "sum"), wape_sum=("wape_w", "sum"),
        wsum=("w_valid", "sum"), folds=("fold", "nunique"),
        mase=("MASE", "mean"), mae=("MAE", "mean"), mape=("MAPE", "mean"),
        rel_bias=("rel_bias", "mean"), tier=("tier", "first"),
    ).reset_index()
    denom = agg["wsum"].replace(0, np.nan)
    agg["score"] = agg["score_sum"] / denom
    agg["wape"] = agg["wape_sum"] / denom
    agg = agg.dropna(subset=["score"])
    if agg.empty:
        return pd.DataFrame()

    # Models that could not be tested on every fold rest on weaker evidence, so
    # they carry a small penalty - but they must NOT be excluded outright.
    # Excluding them silently locked every seasonal model out of the whole
    # catalogue: the earliest fold has only 24 months of training data, while
    # Holt-Winters needs 25 and SARIMA needs 28, so all seven seasonal models
    # scored nothing on that fold and lost by default - despite SARIMA ranking
    # best of ~24 methods on the dense products in the fold it could run on.
    max_folds = agg["folds"].max()
    penalty = bt.get("incomplete_fold_penalty", 0.10)
    allow_incomplete = bt.get("allow_incomplete_fold_models", False)
    agg["complete"] = (agg["folds"] >= max_folds).astype(int)
    agg["adj_score"] = agg["score"] * np.where(agg["complete"] == 1, 1.0, 1.0 + penalty)

    agg["is_baseline"] = agg["model"].isin(baselines | {"zero"}).astype(int)

    picks = []
    for item_id, g in agg.groupby("item_id", sort=False, observed=True):
        g = g.sort_values("adj_score")
        if not allow_incomplete:
            # Default. A model tested on fewer folds is NOT comparable to one
            # tested on all of them: the folds differ enormously in difficulty
            # (the crash year scores ~60% worse than the rebound year for every
            # method), so skipping a fold is a free pass on the hard exam.
            # Measured: allowing them degraded the most recent fold from -19.2%
            # to -21.7% total error and WAPE 0.636 -> 0.659.
            gc = g[g["complete"] == 1]
            if not gc.empty:
                g = gc
        best = g.iloc[0]
        base = g[g["is_baseline"] == 1]
        best_base = base.iloc[0] if not base.empty else None

        chosen = best
        reason = "best back-test score"
        if best["complete"] == 0:
            reason = ("best back-test score - tested on the most recent year only, "
                      "so it had to clear a higher bar")
        if best_base is not None and best["is_baseline"] == 0:
            # Require a clear win over the simple option.
            if not (best["adj_score"] < best_base["adj_score"] * (1 - margin)):
                chosen = best_base
                reason = (f"simple method kept - the best complex model did not beat "
                          f"it by the required {margin:.0%} margin")
        picks.append({
            "item_id": item_id,
            "model": chosen["model"],
            "tier": chosen["tier"],
            "selection_score": chosen["score"],
            "backtest_WAPE": chosen["wape"],
            "backtest_total_error_pct": chosen["rel_bias"] * 100.0,
            "backtest_MASE": chosen["mase"],
            "backtest_MAE": chosen["mae"],
            "backtest_MAPE": chosen["mape"],
            "folds_tested": int(chosen["folds"]),
            "selection_reason": reason,
            "runner_up": best["model"] if chosen["model"] != best["model"] else (
                g.iloc[1]["model"] if len(g) > 1 else ""),
        })

    out = pd.DataFrame(picks)
    out["model_plain"] = out["model"].map(MOD.plain_name)
    return out


def delivered_matrix(preds: dict, selection: pd.DataFrame, item_ids: np.ndarray,
                     h: int, cfg: dict | None = None) -> np.ndarray:
    """The per-product forecast matrix the pipeline would actually deliver for
    one back-test fold: each product's selected model, averaged with its
    runner-up when the ensemble policy is enabled.

    Every reporting path (as-delivered accuracy, benchmark, decision ladder,
    intervals, inventory frontier, reconciliation test) reconstructs delivered
    forecasts through THIS function, so reported accuracy cannot silently
    diverge from what the forecast files contain.
    """
    sel = selection.set_index("item_id")
    model_map = sel["model"].to_dict()
    runner_map = (sel["runner_up"].to_dict()
                  if "runner_up" in sel.columns else {})
    blend = bool(cfg and cfg.get("models", {}).get("ensemble_top_k", 1) >= 2)

    chosen = np.full((len(item_ids), h), np.nan)
    for i, iid in enumerate(item_ids):
        m = model_map.get(iid)
        if not m or m not in preds or np.isnan(preds[m][i]).any():
            continue
        row = preds[m][i]
        if blend and m != "zero":
            r = runner_map.get(iid)
            if (isinstance(r, str) and r and r != "zero" and r in preds
                    and not np.isnan(preds[r][i]).any()):
                row = (row + preds[r][i]) / 2.0
        chosen[i] = row
    return chosen


def accuracy_as_delivered(
    fold_store: dict, selection: pd.DataFrame, panel: pd.DataFrame, cfg: dict,
    fiscal_month_of: dict,
) -> pd.DataFrame:
    """Accuracy of the complete delivered pipeline, safeguards included.

    `overall_accuracy` scores the raw model output. This scores what the
    business actually receives: chosen model, then the shortage floor and the
    surplus cap. It is the honest headline number, because those safeguards
    change the answer for roughly 1,300 products.
    """
    from .features import seasonal_profile
    from .forecast import apply_safeguards

    if not fold_store or selection.empty:
        return pd.DataFrame()

    item_ids = panel.index.to_numpy()
    rows = []

    for fold_name, store in fold_store.items():
        preds, actual = store["preds"], store["actual"]
        h = actual.shape[1]
        chosen = delivered_matrix(preds, selection, item_ids, h, cfg)
        valid = ~np.isnan(chosen).any(axis=1)
        if valid.sum() == 0:
            continue

        train_vals = store["train_values"]
        train_panel = pd.DataFrame(train_vals, index=item_ids,
                                   columns=panel.columns[: store["split"]])
        seas = seasonal_profile(train_panel, fiscal_month_of)
        hist_fms = np.array([fiscal_month_of[m] for m in train_panel.columns])

        raw = chosen[valid].copy()
        guarded, floor_f, cap_f = apply_safeguards(
            chosen[valid].copy(), train_vals[valid], hist_fms, seas, cfg)
        av = actual[valid]
        denom = np.abs(av).sum()
        if denom <= 0:
            continue

        rows.append({
            "Fold": fold_name,
            "Test year": f"FY{store['test_fy']}",
            "Products scored": int(valid.sum()),
            "Actual units": float(av.sum()),
            "Forecast units (models only)": float(raw.sum()),
            "Forecast units (as delivered)": float(guarded.sum()),
            "Total error % (models only)": (raw.sum() - av.sum()) / av.sum() * 100,
            "Total error % (as delivered)": (guarded.sum() - av.sum()) / av.sum() * 100,
            "WAPE (models only)": float(np.abs(av - raw).sum() / denom),
            "WAPE (as delivered)": float(np.abs(av - guarded).sum() / denom),
            "Shortage floor applied to": int(floor_f.sum()),
            "Surplus cap applied to": int(cap_f.sum()),
        })
    return pd.DataFrame(rows)


def benchmark_comparison(
    fold_store: dict, selection: pd.DataFrame, panel: pd.DataFrame, cfg: dict,
    fiscal_month_of: dict,
) -> pd.DataFrame:
    """Compare the delivered forecast against the simplest possible alternatives.

    An error figure on its own is not interpretable. "WAPE 1.05" means nothing
    until you know that repeating last year scored 1.49 on the same data. This
    is the comparison that tells a reader whether the method earned its
    complexity, and it is the honest way to present a year in which every
    method performed poorly.
    """
    from .features import seasonal_profile
    from .forecast import apply_safeguards

    if not fold_store or selection.empty:
        return pd.DataFrame()

    item_ids = panel.index.to_numpy()
    rows = []

    for fold_name, store in fold_store.items():
        actual = store["actual"]
        train = store["train_values"]
        h = actual.shape[1]
        if train.shape[1] < h:
            continue

        # --- what this pipeline delivered -------------------------------
        chosen = delivered_matrix(store["preds"], selection, item_ids, h, cfg)
        valid = ~np.isnan(chosen).any(axis=1)
        if valid.sum() == 0:
            continue

        train_panel = pd.DataFrame(train, index=item_ids,
                                   columns=panel.columns[: store["split"]])
        seas = seasonal_profile(train_panel, fiscal_month_of)
        hist_fms = np.array([fiscal_month_of[c] for c in train_panel.columns])
        delivered, _, _ = apply_safeguards(
            chosen[valid].copy(), train[valid], hist_fms, seas, cfg)

        a = actual[valid]
        tv = train[valid]
        # --- the two dumbest possible baselines -------------------------
        seasonal_naive_fc = tv[:, -h:]                       # repeat last year
        flat_fc = np.tile(tv[:, -h:].sum(axis=1, keepdims=True) / h, (1, h))

        denom = np.abs(a).sum()
        if denom <= 0:
            continue

        def _row(name, pred, is_ours=False):
            return {
                "Fold": fold_name,
                "Test year": f"FY{store['test_fy']}",
                "Method": name,
                "WAPE": float(np.abs(a - pred).sum() / denom),
                "Total error %": float((pred.sum() - a.sum()) / a.sum() * 100),
                "_ours": is_ours,
            }

        rows.append(_row("Repeat last year (seasonal naive)", seasonal_naive_fc))
        rows.append(_row("Last year's total, spread evenly", flat_fc))
        rows.append(_row("This forecasting system", delivered, is_ours=True))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # How much better is ours than the best baseline, per fold?
    for fold, g in out.groupby("Fold"):
        best_base = g[~g["_ours"]]["WAPE"].min()
        ours = g[g["_ours"]]["WAPE"].iloc[0]
        out.loc[g.index, "_improvement"] = (best_base - ours) / best_base * 100
    return out


def accuracy_by_level(
    fold_store: dict, selection: pd.DataFrame, panel: pd.DataFrame, cfg: dict,
    fiscal_month_of: dict, items: pd.DataFrame,
) -> pd.DataFrame:
    """The same delivered forecast, scored at every level a decision is made at.

    A single product-by-month error is the hardest measurement that exists and
    is not the level at which any stock or budget decision is actually taken.
    Aggregation cancels offsetting errors, so accuracy improves as the question
    coarsens - that is a property of the mathematics, not presentation. Showing
    the whole ladder, rather than one rung, lets each figure be matched to the
    decision it supports.
    """
    from .features import seasonal_profile
    from .forecast import apply_safeguards

    if not fold_store or selection.empty:
        return pd.DataFrame()

    store = max(fold_store.values(), key=lambda s: s["test_fy"])
    actual, train = store["actual"], store["train_values"]
    h = actual.shape[1]
    item_ids = panel.index.to_numpy()
    chosen = delivered_matrix(store["preds"], selection, item_ids, h, cfg)
    valid = ~np.isnan(chosen).any(axis=1)
    if valid.sum() == 0:
        return pd.DataFrame()

    tp = pd.DataFrame(train, index=item_ids, columns=panel.columns[: store["split"]])
    seas = seasonal_profile(tp, fiscal_month_of)
    hfm = np.array([fiscal_month_of[c] for c in tp.columns])
    F, _, _ = apply_safeguards(chosen[valid].copy(), train[valid], hfm, seas, cfg)
    A = actual[valid]
    vid = item_ids[valid]

    def wape(a, f):
        d = np.abs(np.asarray(a, float)).sum()
        return float(np.abs(np.asarray(a, float) - np.asarray(f, float)).sum() / d) if d else np.nan

    def by_quarter(X):
        keep = (X.shape[1] // 3) * 3
        return X[:, :keep].reshape(X.shape[0], -1, 3).sum(axis=2)

    rows = [
        ("Individual product, individual month", wape(A, F),
         "Reordering a single product for a single month"),
        ("Individual product, quarter", wape(by_quarter(A), by_quarter(F)),
         "Quarterly purchasing for a single product"),
        ("Individual product, full year", wape(A.sum(1), F.sum(1)),
         "Annual volume commitment for a single product"),
    ]
    if not items.empty and "category_1" in items.columns:
        cat = items.set_index("item_id")["category_1"].reindex(vid).fillna("Unknown")
        cA = pd.DataFrame(A, index=cat.to_numpy()).groupby(level=0).sum().to_numpy()
        cF = pd.DataFrame(F, index=cat.to_numpy()).groupby(level=0).sum().to_numpy()
        rows += [
            ("Product category, month", wape(cA, cF),
             "Monthly capacity and warehouse planning per category"),
            ("Product category, quarter", wape(by_quarter(cA), by_quarter(cF)),
             "Quarterly category budgets"),
        ]
    rows += [
        ("Whole company, month", wape(A.sum(0), F.sum(0)),
         "Monthly cash-flow and headcount planning"),
        ("Whole company, full year", wape(A.sum(), F.sum()),
         "The annual revenue budget"),
    ]
    out = pd.DataFrame(rows, columns=["Decision level", "WAPE", "Decision supported"])
    out["Accuracy %"] = ((1 - out["WAPE"]).clip(lower=0) * 100).round(1)
    out["Test year"] = f"FY{store['test_fy']}"
    return out[["Decision level", "Decision supported", "WAPE", "Accuracy %", "Test year"]]


def overall_accuracy(results: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """Portfolio accuracy of the SELECTED models, per fold - the headline number."""
    if results.empty or selection.empty:
        return pd.DataFrame()
    sel = selection[["item_id", "model"]]
    picked = results.merge(sel, on=["item_id", "model"], how="inner")

    out = []
    for fold, g in picked.groupby("fold", observed=True):
        act, fc = g["actual_total"].sum(), g["forecast_total"].sum()
        # Volume-weighted error across the portfolio.
        # MAE is per month; scale by the fold's actual window length rather
        # than a hardcoded 12, so a changed horizon cannot silently corrupt WAPE.
        num = (g["MAE"] * g["n_months"]).sum()
        out.append({
            "Fold": fold,
            "Test year": f"FY{int(g['test_fy'].iloc[0])}",
            "Products scored": g["item_id"].nunique(),
            "Actual units": act,
            "Forecast units": fc,
            "Total error (units)": fc - act,
            "Total error %": (fc - act) / act * 100 if act else np.nan,
            "WAPE": num / act if act else np.nan,
            "Median product WAPE": g["WAPE"].replace([np.inf], np.nan).median(),
            "Median product MASE": g["MASE"].median(),
            "Median product MAPE": g["MAPE"].median(),
        })
    return pd.DataFrame(out)
