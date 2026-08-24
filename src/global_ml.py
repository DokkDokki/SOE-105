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

# Engineered features (2026-08 experiment, docs/IDEAS.md I): improved both
# folds AND the honest select-on-fold-1/measure-on-fold-2 test. Origin-level
# features describe the product at the time the forecast is made; target-level
# features describe the specific future month being predicted.
EXT_ORIGIN = [
    # demand shape
    "r3_over_r12", "lag1_over_r12", "cv", "max12_over_r12", "spikes_24",
    # customer side - breadth, concentration and mix of the buyers
    "cust12", "conc12", "vip_share12", "cust_mean12", "qty_per_line12",
    "lines12",
    # item master
    "pstat", "lp_zero",
]
EXT_TARGET = [
    "smly_target", "sm2y_target", "own_si_target", "cat_si_target",
    "h", "since_plus_h",
]


class AuxPanels:
    """Customer/order-line panels for the engineered features.

    Built ONCE from the cleaned sales lines and reused by every fold: all
    features are computed per time origin `t` using only columns `<= t`, so a
    full-length panel is leakage-safe for a training window that is a prefix
    of the month axis (which both the back-test and the final fit are).
    """

    def __init__(self, lines, custm, vipq, cust12, conc12, pstat, lp_zero):
        self.lines, self.custm, self.vipq = lines, custm, vipq
        self.cust12, self.conc12 = cust12, conc12
        self.pstat, self.lp_zero = pstat, lp_zero

    @classmethod
    def build(cls, sales: pd.DataFrame, customers: pd.DataFrame,
              items: pd.DataFrame, item_index: pd.Index, months: list) -> "AuxPanels":
        n_items, n_months = len(item_index), len(months)
        dem = sales[sales["is_demand"]] if "is_demand" in sales.columns else sales
        dem = dem.copy()
        dem["mi"] = dem["month"].map({m: i for i, m in enumerate(months)})
        dem["ri"] = dem["item_id"].map({iid: i for i, iid in enumerate(item_index)})
        dem = dem.dropna(subset=["mi", "ri"])
        dem["mi"] = dem["mi"].astype(int)
        dem["ri"] = dem["ri"].astype(int)

        if customers is not None and "customer_type" in customers.columns:
            ct = customers.set_index("customer_id")["customer_type"]
            dem["_vip"] = dem["customer_id"].map(ct).eq("VIP").fillna(False)
        else:
            dem["_vip"] = False

        def mat(df, col=None, agg="size"):
            g = df.groupby(["ri", "mi"])
            s = (g.size() if agg == "size"
                 else g[col].nunique() if agg == "nunique" else g[col].sum())
            M = np.zeros((n_items, n_months))
            if len(s):
                M[s.index.get_level_values(0), s.index.get_level_values(1)] = \
                    s.to_numpy(dtype=float)
            return M

        lines = mat(dem)
        custm = mat(dem, "customer_id", "nunique")
        vipq = mat(dem[dem["_vip"]], "quantity", "sum")

        # Distinct customers and top-customer share over the trailing 12
        # months, per origin month (indexed [t, product]).
        pairs = dem.groupby(["ri", "customer_id", "mi"])["quantity"].sum().reset_index()
        cust12 = np.zeros((n_months, n_items))
        conc12 = np.zeros((n_months, n_items))
        for t in range(n_months):
            w = pairs[(pairs["mi"] <= t) & (pairs["mi"] > t - 12)]
            if w.empty:
                continue
            per = w.groupby(["ri", "customer_id"])["quantity"].sum()
            nc = per.groupby(level=0).size()
            tot = per.groupby(level=0).sum()
            mx = per.groupby(level=0).max()
            cust12[t, nc.index.to_numpy()] = nc.to_numpy(dtype=float)
            conc12[t, tot.index.to_numpy()] = \
                (mx / tot.replace(0, np.nan)).fillna(0).to_numpy(dtype=float)

        attrs = items.set_index("item_id") if items is not None and not items.empty \
            else pd.DataFrame()
        ps = pd.Series(item_index).map(
            attrs["production_status"] if "production_status" in attrs.columns
            else pd.Series(dtype=object))
        pstat = ps.astype("category").cat.codes.to_numpy(dtype=float)
        lp = pd.Series(item_index).map(
            attrs["list_price"] if "list_price" in attrs.columns
            else pd.Series(dtype=float)).fillna(0)
        lp_zero = (lp <= 0).to_numpy(dtype=float)
        return cls(lines, custm, vipq, cust12, conc12, pstat, lp_zero)


class GlobalMLForecaster:
    """Pooled gradient-boosting forecaster over a product x month panel."""

    def __init__(self, horizon: int = 12, random_state: int = 42,
                 quantile: float | None = None, recency_halflife: float | None = 24.0,
                 aux: "AuxPanels | None" = None):
        self.horizon = horizon
        self.random_state = random_state
        self.quantile = quantile           # None = mean forecast (Poisson)
        self.recency_halflife = recency_halflife
        self.aux = aux                     # engineered features on, when given
        self.models: dict[int, HistGradientBoostingRegressor] = {}
        if aux is not None:
            self.feature_names = (list(FEATURE_NAMES[:-1]) + EXT_ORIGIN
                                  + ["target_fiscal_month"] + EXT_TARGET)
        else:
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

        base = np.column_stack([
            lag(1), lag(2), lag(3), lag(6), lag(12),
            roll(3), roll(6), roll(12),
            nz_rate(3), nz_rate(6), nz_rate(12),
            hist.mean(axis=1), hist.std(axis=1), months_since, age, active,
            cat_lag_1[grp], cat_roll_6[grp], cat_roll_12[grp], cat_yoy[grp], share,
            np.full(n, mkt_lag_1), np.full(n, mkt_roll_6), np.full(n, mkt_yoy),
            np.full(n, fiscal_months[t]), np.full(n, float(t)),
            static,
        ])
        if self.aux is None:
            return base

        # ---------------- engineered origin-level features ----------------
        aux = self.aux
        r12 = hist[:, -min(12, m):].mean(axis=1)
        r3 = hist[:, -min(3, m):].mean(axis=1)
        lag1 = hist[:, -1]
        safe = np.where(r12 > 0, r12, 1.0)
        mean_all = hist.mean(axis=1)
        std_all = hist.std(axis=1)
        cv = np.where(mean_all > 0, std_all / np.where(mean_all > 0, mean_all, 1), 0.0)
        max12 = hist[:, -min(12, m):].max(axis=1)
        w24 = hist[:, -min(24, m):]
        med_nz = np.array([np.median(r[r > 0]) if (r > 0).any() else 0.0 for r in w24])
        spikes = ((w24 > 2 * np.maximum(med_nz, 1e-9)[:, None])
                  & (w24 > 2 * r12[:, None])).sum(axis=1)
        own12 = hist[:, -min(12, m):].sum(axis=1)
        vipq = aux.vipq[:, max(0, t - 11):t + 1].sum(axis=1)
        vip_share = np.where(own12 > 0, vipq / np.where(own12 > 0, own12, 1), 0.0)
        cust_mean = aux.custm[:, max(0, t - 11):t + 1].mean(axis=1)
        lines12 = aux.lines[:, max(0, t - 11):t + 1].sum(axis=1)
        qpl = np.where(lines12 > 0, own12 / np.where(lines12 > 0, lines12, 1), 0.0)
        ext = np.column_stack([
            r3 / safe, lag1 / safe, cv, max12 / safe, spikes,
            aux.cust12[t], aux.conc12[t], vip_share, cust_mean, qpl, lines12,
            aux.pstat, aux.lp_zero,
        ])
        return np.column_stack([base, ext])

    def _target_block(self, values, t, h, fiscal_months, target_fm, grp, G, n):
        """Features describing the specific TARGET month `t + h`.

        Uses only values[:, :t+1] (h <= 12 keeps every index <= t)."""
        cols = [np.full(n, float(target_fm))]
        if self.aux is None:
            return np.column_stack(cols)
        hist = values[:, :t + 1]
        j1, j2 = t + h - 12, t + h - 24
        smly = values[:, j1] if j1 >= 0 else np.zeros(n)
        sm2y = values[:, j2] if j2 >= 0 else smly
        sel = fiscal_months[:t + 1] == target_fm
        own_mean = hist.mean(axis=1)
        own_fm = hist[:, sel].mean(axis=1) if sel.any() else own_mean
        own_si = np.where(own_mean > 0, own_fm / np.where(own_mean > 0, own_mean, 1), 1.0)
        cat_hist = G @ hist
        cat_mean = cat_hist.mean(axis=1)
        cat_fm = cat_hist[:, sel].mean(axis=1) if sel.any() else cat_mean
        cat_si = np.where(cat_mean > 0,
                          cat_fm / np.where(cat_mean > 0, cat_mean, 1), 1.0)[grp]
        nz = hist > 0
        any_sale = nz.any(axis=1)
        since = np.where(any_sale, nz[:, ::-1].argmax(axis=1), 99)
        cols += [smly, sm2y, own_si, cat_si, np.full(n, float(h)), since + h]
        return np.column_stack(cols)

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

        n = values.shape[0]
        for h in range(1, self.horizon + 1):
            X_parts, y_parts, w_parts = [], [], []
            for t in range(MIN_ORIGIN, n_months - h):
                block = blocks[t]
                tb = self._target_block(values, t, h, fiscal_months,
                                        fiscal_months[t + h], grp, G, n)
                X_parts.append(np.column_stack([block, tb]))
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
        n = values.shape[0]
        out = np.zeros((len(panel), self.horizon))
        for h in range(1, self.horizon + 1):
            model = self.models.get(h)
            if model is None:
                out[:, h - 1] = base[:, roll6]
                continue
            tb = self._target_block(values, t, h, fiscal_months,
                                    float(future_fiscal_months[h - 1]), grp, G, n)
            X = np.column_stack([base, tb])
            out[:, h - 1] = np.clip(model.predict(X), 0, None)
        return pd.DataFrame(out, index=panel.index)
