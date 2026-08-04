"""A single machine-learning model trained across ALL products at once.

TERMINOLOGY WARNING
    In forecasting, a "global" model means one model fitted across ALL the
    series in the dataset, as opposed to a "local" model fitted separately to
    each series. It has nothing to do with global/world data, and NO data from
    outside the supplied files is used anywhere in this project. Every number
    this model sees comes from the company's own sales history, item master and
    customer master. There are no internet calls, no market indices, no
    third-party data. "All products" means all products in THIS dataset.


Why this and not a separate ML model per product: most products here have only
a handful of observations, far too few to train anything on individually. A
"global" model pools every product's history, so a sparse product borrows
strength from the thousands of products that behave like it. This is the
standard modern approach to large, sparse retail-style catalogues.

Strategy is DIRECT multi-horizon: a separate regressor is trained for each of
the 12 months ahead, which avoids the error accumulation of feeding a model its
own predictions back in.

CROSS-SERIES FEATURES
    The whole point of pooling products into one model is to let them share
    information. An earlier version only fed each product its OWN lags, which
    wasted that advantage entirely and left the model ranking below average on
    the dense products. It now also sees, at every point in time and using only
    data available at that point:
      * how its product CATEGORY is doing (level, trend, year-on-year)
      * how the WHOLE market is doing (level, trend, year-on-year)
      * its own share of its category
    Category and market series are dense and well behaved even when an
    individual product is not, so this is where a sparse product gets the
    context it cannot get from its own three data points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SEASON = 12
MIN_ORIGIN = 12          # need 12 months of history before an origin is usable

FEATURE_NAMES = [
    # --- the product's own history ---
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6", "roll_mean_12",
    "nonzero_rate_3", "nonzero_rate_6", "nonzero_rate_12",
    "hist_mean", "hist_std", "months_since_last_sale", "age_months",
    "active_months",
    # --- cross-series: the product's category ---
    "cat_lag_1", "cat_roll_6", "cat_roll_12", "cat_yoy", "share_of_cat_12",
    # --- cross-series: the whole market ---
    "mkt_lag_1", "mkt_roll_6", "mkt_yoy",
    # --- calendar and attributes ---
    "origin_fiscal_month", "t",
    "cat1_code", "cat2_code", "log_list_price",
    "target_fiscal_month",
]
# Tree-native categorical columns (positions in FEATURE_NAMES).
CATEGORICAL = ["cat1_code", "cat2_code"]


class GlobalMLForecaster:
    """Pooled gradient-boosting forecaster over a product x month panel."""

    def __init__(self, horizon: int = 12, random_state: int = 42,
                 quantile: float | None = None, recency_halflife: float | None = 24.0):
        self.horizon = horizon
        self.random_state = random_state
        self.quantile = quantile           # None = mean forecast (Poisson)
        self.recency_halflife = recency_halflife
        self.models: dict[int, HistGradientBoostingRegressor] = {}
        self.feature_names = list(FEATURE_NAMES)
        self._fitted = False

    # ---------------------------------------------------------------- static
    @staticmethod
    def _static_features(item_ids: np.ndarray, items: pd.DataFrame):
        """Product attributes plus the group index used for category aggregates."""
        n = len(item_ids)
        codes1 = np.zeros(n)
        codes2 = np.zeros(n)
        logp = np.zeros(n)
        if items is not None and not items.empty:
            attrs = items.set_index("item_id")
            if "category_1" in attrs.columns:
                s = pd.Series(item_ids).map(attrs["category_1"]).astype("category")
                codes1 = s.cat.codes.to_numpy(dtype=float)
            if "category_2" in attrs.columns:
                s = pd.Series(item_ids).map(attrs["category_2"]).astype("category")
                codes2 = s.cat.codes.to_numpy(dtype=float)
            if "list_price" in attrs.columns:
                lp = pd.Series(item_ids).map(attrs["list_price"]).fillna(0).clip(lower=0)
                logp = np.log1p(lp).to_numpy(dtype=float)
        # Group index for category aggregates (never negative).
        grp = np.where(codes1 >= 0, codes1, 0).astype(int)
        return np.column_stack([codes1, codes2, logp]), grp

    @staticmethod
    def _group_matrix(grp: np.ndarray) -> np.ndarray:
        """Dense indicator matrix (n_groups x n_products) for fast aggregation."""
        n_groups = int(grp.max()) + 1 if len(grp) else 1
        G = np.zeros((n_groups, len(grp)))
        G[grp, np.arange(len(grp))] = 1.0
        return G

    # -------------------------------------------------------------- features
    def _origin_features(
        self, values: np.ndarray, t: int, fiscal_months: np.ndarray,
        static: np.ndarray, grp: np.ndarray, G: np.ndarray,
    ) -> np.ndarray:
        """Feature block for every product at a single time origin `t`.

        Only `values[:, :t+1]` is ever touched - nothing later. That is what
        keeps the back-test honest.
        """
        hist = values[:, : t + 1]
        n, m = hist.shape

        def lag(k):
            return hist[:, -k] if m >= k else np.zeros(n)

        def roll(k):
            return hist[:, -min(k, m):].mean(axis=1)

        def nz_rate(k):
            w = hist[:, -min(k, m):]
            return (w > 0).mean(axis=1)

        nz = hist > 0
        any_sale = nz.any(axis=1)
        last_sale_idx = np.where(any_sale, m - 1 - nz[:, ::-1].argmax(axis=1), -1)
        months_since = np.where(any_sale, m - 1 - last_sale_idx, 99)
        first_sale_idx = np.where(any_sale, nz.argmax(axis=1), m)
        age = np.where(any_sale, m - first_sale_idx, 0)
        active = nz.sum(axis=1)

        # ---------------- cross-series: category level ----------------
        cat_hist = G @ hist                                  # (n_groups x m)
        cat_lag_1 = cat_hist[:, -1] if m >= 1 else np.zeros(cat_hist.shape[0])
        cat_roll_6 = cat_hist[:, -min(6, m):].mean(axis=1)
        cat_roll_12 = cat_hist[:, -min(12, m):].mean(axis=1)
        if m >= 24:
            last12 = cat_hist[:, -12:].sum(axis=1)
            prev12 = cat_hist[:, -24:-12].sum(axis=1)
            cat_yoy = np.where(prev12 > 0, last12 / np.maximum(prev12, 1e-9), 1.0)
        else:
            cat_yoy = np.ones(cat_hist.shape[0])
        cat_last12 = cat_hist[:, -min(12, m):].sum(axis=1)

        own_last12 = hist[:, -min(12, m):].sum(axis=1)
        denom = cat_last12[grp]
        share = np.where(denom > 0, own_last12 / np.maximum(denom, 1e-9), 0.0)

        # ---------------- cross-series: whole market ----------------
        mkt = hist.sum(axis=0)                               # (m,)
        mkt_lag_1 = mkt[-1] if m >= 1 else 0.0
        mkt_roll_6 = mkt[-min(6, m):].mean()
        if m >= 24:
            mkt_yoy = mkt[-12:].sum() / max(mkt[-24:-12].sum(), 1e-9)
        else:
            mkt_yoy = 1.0

        return np.column_stack([
            lag(1), lag(2), lag(3), lag(6), lag(12),
            roll(3), roll(6), roll(12),
            nz_rate(3), nz_rate(6), nz_rate(12),
            hist.mean(axis=1), hist.std(axis=1), months_since, age, active,
            cat_lag_1[grp], cat_roll_6[grp], cat_roll_12[grp], cat_yoy[grp], share,
            np.full(n, mkt_lag_1), np.full(n, mkt_roll_6), np.full(n, mkt_yoy),
            np.full(n, fiscal_months[t]), np.full(n, float(t)),
            static,
        ])

    def _build_origins(self, values, fiscal_months, static, grp, G, n_months, max_h):
        """Origin feature blocks, computed once and reused by all 12 horizons."""
        return {t: self._origin_features(values, t, fiscal_months, static, grp, G)
                for t in range(MIN_ORIGIN, n_months - 1)}

    # ------------------------------------------------------------------- fit
    def fit(self, panel: pd.DataFrame, items: pd.DataFrame, fiscal_month_of: dict,
            sample_mask: np.ndarray | None = None) -> "GlobalMLForecaster":
        if panel.empty or panel.shape[1] <= MIN_ORIGIN + 1:
            self._fitted = False
            return self

        values = panel.to_numpy(dtype=float)
        months = list(panel.columns)
        fiscal_months = np.array([fiscal_month_of[m] for m in months], dtype=float)
        static, grp = self._static_features(panel.index.to_numpy(), items)
        G = self._group_matrix(grp)
        n_months = values.shape[1]

        blocks = self._build_origins(values, fiscal_months, static, grp, G,
                                     n_months, self.horizon)
        age_col = self.feature_names.index("age_months")
        cat_idx = [self.feature_names.index(c) for c in CATEGORICAL]
        self.models = {}

        for h in range(1, self.horizon + 1):
            X_parts, y_parts, w_parts = [], [], []
            for t in range(MIN_ORIGIN, n_months - h):
                block = blocks[t]
                target_fm = fiscal_months[t + h]
                X_parts.append(np.column_stack(
                    [block, np.full(block.shape[0], target_fm)]))
                y_parts.append(values[:, t + h])
                if self.recency_halflife:
                    # Older origins mislead after the FY22->FY23 level break.
                    age = (n_months - 1) - (t + h)
                    w_parts.append(np.full(block.shape[0],
                                           0.5 ** (age / self.recency_halflife)))
            if not X_parts:
                continue
            X = np.vstack(X_parts)
            y = np.concatenate(y_parts)
            w = np.concatenate(w_parts) if w_parts else None

            # Drop products that had not started selling yet - structural
            # zeros, not demand signal.
            alive = X[:, age_col] > 0
            if sample_mask is not None:
                alive = alive & np.tile(sample_mask, len(X_parts))
            if alive.sum() > 500:
                X, y = X[alive], y[alive]
                w = w[alive] if w is not None else None
            if len(y) < 200:
                continue

            if self.quantile is not None:
                model = HistGradientBoostingRegressor(
                    loss="quantile", quantile=self.quantile,
                    max_iter=180, learning_rate=0.06, max_depth=6,
                    min_samples_leaf=40, l2_regularization=1.0,
                    early_stopping=True, validation_fraction=0.12,
                    n_iter_no_change=20, random_state=self.random_state,
                    categorical_features=cat_idx,
                )
            else:
                model = HistGradientBoostingRegressor(
                    loss="poisson" if (y >= 0).all() else "squared_error",
                    max_iter=200, learning_rate=0.06, max_depth=6,
                    min_samples_leaf=40, l2_regularization=1.0,
                    early_stopping=True, validation_fraction=0.12,
                    n_iter_no_change=20, random_state=self.random_state,
                    categorical_features=cat_idx,
                )
            model.fit(X, np.clip(y, 0, None), sample_weight=w)
            self.models[h] = model

        self._fitted = len(self.models) > 0
        return self

    # --------------------------------------------------------------- predict
    def predict(self, panel: pd.DataFrame, items: pd.DataFrame,
                fiscal_month_of: dict, future_fiscal_months: list[int]) -> pd.DataFrame:
        """Forecast the next `horizon` months for every product in the panel."""
        if not self._fitted or panel.empty:
            return pd.DataFrame(np.zeros((len(panel), self.horizon)), index=panel.index)

        values = panel.to_numpy(dtype=float)
        months = list(panel.columns)
        fiscal_months = np.array([fiscal_month_of[m] for m in months], dtype=float)
        static, grp = self._static_features(panel.index.to_numpy(), items)
        G = self._group_matrix(grp)
        t = values.shape[1] - 1

        base = self._origin_features(values, t, fiscal_months, static, grp, G)
        roll6 = self.feature_names.index("roll_mean_6")
        out = np.zeros((len(panel), self.horizon))
        for h in range(1, self.horizon + 1):
            model = self.models.get(h)
            if model is None:
                out[:, h - 1] = base[:, roll6]
                continue
            X = np.column_stack([base, np.full(base.shape[0],
                                               float(future_fiscal_months[h - 1]))])
            out[:, h - 1] = np.clip(model.predict(X), 0, None)
        return pd.DataFrame(out, index=panel.index)
