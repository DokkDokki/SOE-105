"""Accuracy metrics.

MAPE is reported because it was asked for, but it is NOT the selection metric.
With this much intermittent demand, MAPE is undefined whenever an actual is
zero and explodes when an actual is near zero, so a MAPE-driven choice would be
dominated by the products that matter least. WAPE (volume-weighted absolute
error) and MASE are used for model selection instead, and every MAPE figure is
published alongside the number of months it had to skip.
"""
from __future__ import annotations

import numpy as np

SEASON = 12


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs(a - p))) if len(a) else np.nan


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((a - p) ** 2))) if len(a) else np.nan


def bias(actual: np.ndarray, pred: np.ndarray) -> float:
    """Positive = we over-forecast (risk of surplus stock).
    Negative = we under-forecast (risk of shortage)."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(p - a)) if len(a) else np.nan


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Weighted absolute percentage error = sum|error| / sum(actual).

    Well defined when individual months are zero, and weights months by their
    volume, which is what inventory planning actually cares about.
    """
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    denom = np.abs(a).sum()
    if denom <= 0:
        # No demand at all in the test window: perfect only if we also said zero.
        return 0.0 if np.abs(p).sum() == 0 else np.inf
    return float(np.abs(a - p).sum() / denom)


def mape(actual: np.ndarray, pred: np.ndarray) -> tuple[float, int]:
    """Classic MAPE, plus the count of months it could not be computed on."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    mask = a != 0
    skipped = int((~mask).sum())
    if mask.sum() == 0:
        return np.nan, skipped
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100.0), skipped


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric MAPE - defined even when the actual is zero."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    denom = (np.abs(a) + np.abs(p)) / 2.0
    mask = denom > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(a[mask] - p[mask]) / denom[mask]) * 100.0)


def mase(actual: np.ndarray, pred: np.ndarray, train: np.ndarray,
         seasonal: bool = True) -> float:
    """Mean absolute scaled error.

    1.0 means "no better than the naive benchmark"; below 1.0 means the model
    genuinely adds value. This is the only metric here that is comparable
    across products of wildly different volume.
    """
    a, p, tr = (np.asarray(x, float) for x in (actual, pred, train))
    m = SEASON if seasonal and len(tr) > SEASON else 1
    if len(tr) <= m:
        return np.nan
    scale = np.mean(np.abs(tr[m:] - tr[:-m]))
    if scale <= 0:
        scale = np.mean(np.abs(np.diff(tr))) if len(tr) > 1 else 0.0
    if scale <= 0:
        return np.nan
    return float(np.mean(np.abs(a - p)) / scale)


def score_all(actual, pred, train) -> dict:
    """Every metric for one product-fold."""
    mp, skipped = mape(actual, pred)
    return {
        "MAE": mae(actual, pred),
        "RMSE": rmse(actual, pred),
        "WAPE": wape(actual, pred),
        "MAPE": mp,
        "MAPE_months_skipped": skipped,
        "SMAPE": smape(actual, pred),
        "MASE": mase(actual, pred, train),
        "Bias": bias(actual, pred),
        "actual_total": float(np.sum(actual)),
        "forecast_total": float(np.sum(pred)),
    }


def aggregate_scores(actual_matrix: np.ndarray, pred_matrix: np.ndarray) -> dict:
    """Portfolio-level accuracy across all products at once."""
    a, p = np.asarray(actual_matrix, float), np.asarray(pred_matrix, float)
    total_abs_err = np.abs(a - p).sum()
    total_actual = np.abs(a).sum()
    return {
        "overall_WAPE": float(total_abs_err / total_actual) if total_actual > 0 else np.nan,
        "overall_MAE": float(np.abs(a - p).mean()),
        "overall_bias": float((p - a).sum()),
        "total_actual": float(a.sum()),
        "total_forecast": float(p.sum()),
    }
