"""The candidate forecasting methods.

Every model shares one interface:

    forecast(y, h, ctx) -> np.ndarray of length h

where `y` is the product's monthly demand history (including genuine zeros),
`h` is the number of months to forecast, and `ctx` carries shared context such
as the fiscal-month of each historical point and a pooled seasonal profile.

No model is assumed to be right. Which one is used for a given product is
decided by back-testing in backtest.py.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

warnings.filterwarnings("ignore")

SEASON = 12


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _clean(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def _nonneg(x: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def _recency_weights(n: int, halflife: float) -> np.ndarray:
    """Exponential weights so recent months count more.

    This matters here: the business went through a sharp level break between
    FY2022 and FY2023, so older months are a poor guide to the current level.
    """
    if halflife is None or halflife <= 0:
        return np.ones(n)
    age = np.arange(n - 1, -1, -1, dtype=float)
    return 0.5 ** (age / halflife)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def zero(y, h, ctx=None):
    return np.zeros(h)


def naive(y, h, ctx=None):
    y = _clean(y)
    return _nonneg(np.repeat(y[-1] if len(y) else 0.0, h))


def mean_all(y, h, ctx=None):
    y = _clean(y)
    return _nonneg(np.repeat(y.mean() if len(y) else 0.0, h))


def _moving_average(y, h, window):
    y = _clean(y)
    if len(y) == 0:
        return np.zeros(h)
    return _nonneg(np.repeat(y[-min(window, len(y)):].mean(), h))


def moving_average_3(y, h, ctx=None):
    return _moving_average(y, h, 3)


def moving_average_6(y, h, ctx=None):
    return _moving_average(y, h, 6)


def moving_average_12(y, h, ctx=None):
    return _moving_average(y, h, 12)


def weighted_mean(y, h, ctx=None):
    """Recency-weighted average level - robust to the FY22/FY23 level break."""
    y = _clean(y)
    if len(y) == 0:
        return np.zeros(h)
    hl = (ctx or {}).get("halflife", 18)
    w = _recency_weights(len(y), hl)
    return _nonneg(np.repeat(np.average(y, weights=w), h))


def seasonal_naive(y, h, ctx=None):
    """Repeat the same month of the previous year."""
    y = _clean(y)
    if len(y) < SEASON:
        return mean_all(y, h)
    last = y[-SEASON:]
    return _nonneg(np.array([last[i % SEASON] for i in range(h)]))


def seasonal_naive_avg2(y, h, ctx=None):
    """Average the same month across the last two years - less noisy."""
    y = _clean(y)
    if len(y) < 2 * SEASON:
        return seasonal_naive(y, h)
    a, b = y[-SEASON:], y[-2 * SEASON:-SEASON]
    avg = (a + b) / 2.0
    return _nonneg(np.array([avg[i % SEASON] for i in range(h)]))


def drift(y, h, ctx=None):
    """Straight line through the first and last observation, damped."""
    y = _clean(y)
    if len(y) < 2:
        return naive(y, h)
    slope = (y[-1] - y[0]) / (len(y) - 1)
    # Damp the slope so a 12-month extrapolation cannot run away.
    damp = 0.6
    return _nonneg(y[-1] + slope * damp * np.arange(1, h + 1))


def seasonal_profile_model(y, h, ctx=None):
    """Recent level multiplied by the pooled category seasonal shape.

    For products with a year or two of history, borrowing the seasonal shape
    from the wider category is far more reliable than trying to estimate 12
    seasonal factors from a handful of observations.
    """
    y = _clean(y)
    ctx = ctx or {}
    prof = ctx.get("seasonal_profile")
    fm_future = ctx.get("future_fiscal_months")
    if prof is None or fm_future is None or len(y) == 0:
        return moving_average_6(y, h)

    fm_hist = ctx.get("hist_fiscal_months")
    window = min(len(y), 12)
    recent = y[-window:]
    if fm_hist is not None and len(fm_hist) >= window:
        # Deseasonalise the recent window before taking its level.
        f = np.array([prof[(m - 1) % 12] for m in fm_hist[-window:]], dtype=float)
        f = np.where(f > 1e-6, f, 1.0)
        level = float(np.mean(recent / f))
    else:
        level = float(recent.mean())

    factors = np.array([prof[(m - 1) % 12] for m in fm_future[:h]], dtype=float)
    return _nonneg(level * factors)


# --------------------------------------------------------------------------
# Intermittent demand - Croston and the Syntetos-Boylan Approximation
# --------------------------------------------------------------------------
def _croston_core(y: np.ndarray, alpha: float) -> tuple[float, float]:
    """Return smoothed demand size and smoothed interval between demands."""
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0.0, 1.0
    sizes = y[nz]
    intervals = np.diff(np.concatenate([[-1], nz]))

    z = float(sizes[0])
    p = float(max(intervals[0], 1))
    for i in range(1, len(sizes)):
        z += alpha * (sizes[i] - z)
        p += alpha * (max(intervals[i], 1) - p)
    return z, max(p, 1e-6)


def croston(y, h, ctx=None):
    """Croston's method: the standard approach for intermittent demand."""
    y = _clean(y)
    alpha = (ctx or {}).get("croston_alpha", 0.1)
    z, p = _croston_core(y, alpha)
    return _nonneg(np.repeat(z / p, h))


def croston_sba(y, h, ctx=None):
    """Syntetos-Boylan Approximation - Croston with its known bias removed."""
    y = _clean(y)
    alpha = (ctx or {}).get("croston_alpha", 0.1)
    z, p = _croston_core(y, alpha)
    return _nonneg(np.repeat((1 - alpha / 2.0) * z / p, h))


def tsb(y, h, ctx=None):
    """Teunter-Syntetos-Babai: Croston for products drifting to obsolescence.

    Croston only updates when a sale happens, so a product that has stopped
    selling keeps its last positive forecast for ever. TSB updates the demand
    PROBABILITY every month, including the silent ones, letting the forecast
    decay towards zero as a product goes quiet - the right behaviour for a
    catalogue with 447 discontinued items still trailing off.
    """
    y = _clean(y)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return np.zeros(h)
    alpha = (ctx or {}).get("croston_alpha", 0.1)   # probability smoothing
    beta = 0.1                                       # size smoothing
    start = nz[0]
    prob = 1.0
    size = float(y[start])
    for t in range(start + 1, len(y)):
        if y[t] > 0:
            prob += alpha * (1.0 - prob)
            size += beta * (float(y[t]) - size)
        else:
            prob += alpha * (0.0 - prob)
    return _nonneg(np.repeat(prob * size, h))


def demand_rate(y, h, ctx=None):
    """Simple average demand per month over the product's life so far."""
    y = _clean(y)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return np.zeros(h)
    life = len(y) - nz[0]
    return _nonneg(np.repeat(y[nz[0]:].sum() / max(life, 1), h))


# --------------------------------------------------------------------------
# Exponential smoothing (statsmodels)
# --------------------------------------------------------------------------
def _ets(y, h, trend=None, damped=False, seasonal=None):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = _clean(y)
    need = 2 * SEASON + 1 if seasonal else 3
    if len(y) < need:
        return None
    if seasonal == "mul" and (y <= 0).any():
        return None

    data = y.astype(float)
    if seasonal == "mul" or trend == "mul":
        data = np.where(data <= 0, 1e-3, data)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                data, trend=trend, damped_trend=damped and trend is not None,
                seasonal=seasonal, seasonal_periods=SEASON if seasonal else None,
                initialization_method="estimated",
            ).fit(optimized=True)
            fc = np.asarray(fit.forecast(h), dtype=float)
        if not np.isfinite(fc).all():
            return None
        return _nonneg(fc)
    except Exception:
        return None


def ses(y, h, ctx=None):
    return _ets(y, h, trend=None, damped=False, seasonal=None)


def holt_damped(y, h, ctx=None):
    return _ets(y, h, trend="add", damped=True, seasonal=None)


def hw_add(y, h, ctx=None):
    return _ets(y, h, trend="add", damped=False, seasonal="add")


def hw_add_damped(y, h, ctx=None):
    return _ets(y, h, trend="add", damped=True, seasonal="add")


def hw_seasonal_only(y, h, ctx=None):
    return _ets(y, h, trend=None, damped=False, seasonal="add")


def hw_mul(y, h, ctx=None):
    return _ets(y, h, trend="add", damped=True, seasonal="mul")


# --------------------------------------------------------------------------
# SARIMA (statsmodels)
# --------------------------------------------------------------------------
def _sarima(y, h, order, seasonal_order):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = _clean(y)
    if len(y) < 2 * SEASON + 4:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(
                y.astype(float), order=tuple(order), seasonal_order=tuple(seasonal_order),
                enforce_stationarity=False, enforce_invertibility=False,
                trend=None,
            ).fit(disp=0, maxiter=60)
            fc = np.asarray(fit.forecast(h), dtype=float)
        if not np.isfinite(fc).all():
            return None
        return _nonneg(fc)
    except Exception:
        return None


def make_sarima(order, seasonal_order) -> Callable:
    def _f(y, h, ctx=None):
        return _sarima(y, h, order, seasonal_order)
    _f.__name__ = f"sarima_{''.join(map(str, order))}_{''.join(map(str, seasonal_order))}"
    return _f


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
BASELINE_MODELS: dict[str, Callable] = {
    "naive": naive,
    "mean_all": mean_all,
    "moving_average_3": moving_average_3,
    "moving_average_6": moving_average_6,
    "moving_average_12": moving_average_12,
    "weighted_mean": weighted_mean,
    "seasonal_naive": seasonal_naive,
    "seasonal_naive_avg2": seasonal_naive_avg2,
    "drift": drift,
}

INTERMITTENT_MODELS: dict[str, Callable] = {
    "croston": croston,
    "croston_sba": croston_sba,
    "tsb": tsb,
    "demand_rate": demand_rate,
}

SMOOTHING_MODELS: dict[str, Callable] = {
    "ses": ses,
    "holt_damped": holt_damped,
    "holt_winters_add": hw_add,
    "holt_winters_add_damped": hw_add_damped,
    "holt_winters_seasonal_only": hw_seasonal_only,
    "holt_winters_mul": hw_mul,
}

PROFILE_MODELS: dict[str, Callable] = {
    "category_seasonal_profile": seasonal_profile_model,
}


def build_registry(cfg: dict) -> dict[str, Callable]:
    """Assemble the enabled model registry from configuration."""
    reg: dict[str, Callable] = {}
    reg.update(BASELINE_MODELS)
    reg.update(INTERMITTENT_MODELS)
    reg.update(PROFILE_MODELS)

    mc = cfg.get("models", {})
    if mc.get("enable_holt_winters", True):
        reg.update(SMOOTHING_MODELS)
    if mc.get("enable_sarima", True):
        for order, seasonal_order in mc.get("sarima_orders", []):
            f = make_sarima(order, seasonal_order)
            reg[f.__name__] = f
    return reg


# Which model families each product tier is allowed to use. Fitting a seasonal
# model to a product with six observations produces confident nonsense, so the
# candidate set is restricted to what the history can actually support.
def models_for_tier(tier: str, registry: dict[str, Callable]) -> list[str]:
    from .features import TIER_DENSE, TIER_MODERATE, TIER_SPARSE, TIER_DORMANT

    if tier == TIER_DORMANT:
        return []
    if tier == TIER_DENSE:
        return list(registry.keys())
    if tier == TIER_MODERATE:
        # "Repeat last month" is excluded below Tier A. On a product that sells
        # in only some months, last month is very often zero, so naive quietly
        # forecasts a whole year of nothing for a product that genuinely sells.
        allowed = (set(BASELINE_MODELS) | set(INTERMITTENT_MODELS)
                   | set(PROFILE_MODELS) | {"ses", "holt_damped"}) - {"naive"}
        return [m for m in registry if m in allowed]
    if tier == TIER_SPARSE:
        # Lumpy demand needs rate estimators, not short moving averages: a
        # 3-month average of a product that sells twice a year is almost
        # always zero.
        allowed = set(INTERMITTENT_MODELS) | {
            "moving_average_6", "moving_average_12", "mean_all", "weighted_mean",
        }
        return [m for m in registry if m in allowed]
    return list(BASELINE_MODELS)


# Label for products too new for the tournament (see pipeline.run_pipeline).
TOO_NEW_LABEL = "Too new to back-test (zero + safeguard floor)"

PLAIN_ENGLISH_NAMES = {
    "zero": "No forecast (no recent sales)",
    "naive": "Repeat last month",
    "mean_all": "Long-run average",
    "moving_average_3": "Average of last 3 months",
    "moving_average_6": "Average of last 6 months",
    "moving_average_12": "Average of last 12 months",
    "weighted_mean": "Recency-weighted average",
    "seasonal_naive": "Same month last year",
    "seasonal_naive_avg2": "Same month, averaged over 2 years",
    "drift": "Straight-line trend",
    "croston": "Croston (for irregular demand)",
    "croston_sba": "Croston/SBA (for irregular demand)",
    "tsb": "TSB (irregular demand that may be fading out)",
    "demand_rate": "Average demand rate",
    "ses": "Exponential smoothing",
    "holt_damped": "Exponential smoothing with damped trend",
    "holt_winters_add": "Holt-Winters (trend + seasonality)",
    "holt_winters_add_damped": "Holt-Winters (damped trend + seasonality)",
    "holt_winters_seasonal_only": "Holt-Winters (seasonality only)",
    "holt_winters_mul": "Holt-Winters (multiplicative seasonality)",
    "category_seasonal_profile": "Recent level x category seasonal pattern",
    # "Global" is forecasting jargon for "one model fitted across all the series
    # in the dataset" (as opposed to "local" = one model per series). It does
    # NOT mean global/world data. Nothing outside the supplied files is used,
    # so the user-facing label says so plainly.
    "global_ml": "Machine learning (learns from all products in this dataset together)",
}


def plain_name(model: str) -> str:
    if model in PLAIN_ENGLISH_NAMES:
        return PLAIN_ENGLISH_NAMES[model]
    if model.startswith("sarima"):
        return f"SARIMA ({model.replace('sarima_', '')})"
    return model
