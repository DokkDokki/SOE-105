"""End-to-end pipeline: ingest -> clean -> profile -> back-test -> select ->
forecast -> price -> write outputs.

This is the single entry point used by both the command line (run_pipeline.py)
and the dashboard's "upload new data and re-run" button, so the website and the
CLI can never drift apart.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import backtest as BT
from . import clean as CL
from . import features as FE
from . import fiscal as FI
from . import forecast as FC
from . import ingest as IN
from . import intervals as IV
from . import inventory as INV
from . import models as MOD
from . import pricing as PR
from .config import DATA_RAW, OUTPUTS, ensure_dirs, load_config


@dataclass
class PipelineResult:
    forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_by_product: pd.DataFrame = field(default_factory=pd.DataFrame)
    customer_forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    selection: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest_detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    accuracy_delivered: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark: pd.DataFrame = field(default_factory=pd.DataFrame)
    frontier: pd.DataFrame = field(default_factory=pd.DataFrame)
    accuracy_levels: pd.DataFrame = field(default_factory=pd.DataFrame)
    accuracy_by_tier: pd.DataFrame = field(default_factory=pd.DataFrame)
    interval_multipliers: pd.DataFrame = field(default_factory=pd.DataFrame)
    profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality: pd.DataFrame = field(default_factory=pd.DataFrame)
    history_item_month: pd.DataFrame = field(default_factory=pd.DataFrame)
    history_customer_month: pd.DataFrame = field(default_factory=pd.DataFrame)
    history_total: pd.DataFrame = field(default_factory=pd.DataFrame)
    sales_clean: pd.DataFrame = field(default_factory=pd.DataFrame)
    items: pd.DataFrame = field(default_factory=pd.DataFrame)
    customers: pd.DataFrame = field(default_factory=pd.DataFrame)
    group_forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    reconciliation: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    load_reports: list = field(default_factory=list)


def _noop(*_a, **_k) -> None:
    pass


def run_pipeline(
    data_folder: Path | None = None,
    uploads: dict[str, Any] | None = None,
    config_path: Path | None = None,
    write_outputs: bool = True,
    n_jobs: int = -1,
    on_status: Callable[[str, float], None] | None = None,
) -> PipelineResult:
    """Run everything. `on_status(message, fraction_complete)` drives progress bars."""
    status = on_status or _noop
    t0 = time.time()
    ensure_dirs()
    cfg = load_config(config_path)
    res = PipelineResult()

    # ------------------------------------------------------------ 1. ingest
    status("Reading data files...", 0.02)
    folder = Path(data_folder) if data_folder else DATA_RAW
    tables, reports = IN.load_all(folder, uploads)
    res.load_reports = reports

    sales_raw = tables["sales"]
    if sales_raw.empty:
        raise ValueError(
            "No sales history could be read. Put the sales file in data/raw/ "
            "or upload it in the dashboard."
        )

    items, customers = CL.clean_masters(tables["item"], tables["customer"])

    # -------------------------------------------- 2. detect the fiscal year
    status("Detecting the fiscal year...", 0.06)
    det = FI.detect_fiscal_start_month(sales_raw["order_date"])
    configured = cfg["fiscal"]["start_month"]
    if det.confidence in ("high", "medium") and det.start_month != configured:
        # Trust the data over the config, but say so loudly.
        cfg["fiscal"]["start_month"] = det.start_month
    start_month = cfg["fiscal"]["start_month"]

    # ------------------------------------------------------------- 3. clean
    status("Cleaning and reconciling against the master lists...", 0.10)
    sales, qrep = CL.clean_sales(sales_raw, items, customers, cfg)
    res.quality = qrep.to_frame()
    res.sales_clean = sales
    res.items, res.customers = items, customers

    peak_note = FI.seasonal_peak_evidence(sales, start_month)

    # ---------------------------------------------------- 4. monthly panels
    status("Building the monthly demand history...", 0.14)
    panel = FE.build_monthly_panel(sales, "quantity", "is_demand")
    if panel.empty:
        raise ValueError("No usable demand rows after cleaning.")

    months = list(panel.columns)
    fy_of = {m: int(FI.fiscal_year(pd.Series([m.to_timestamp()]), start_month).iloc[0])
             for m in months}
    fm_of = {m: int(FI.fiscal_month(pd.Series([m.to_timestamp()]), start_month).iloc[0])
             for m in months}

    last_fy = max(fy_of.values())
    forecast_fy = cfg["fiscal"].get("forecast_fy") or (last_fy + 1)
    if forecast_fy <= last_fy:
        forecast_fy = last_fy + 1
    horizon = cfg["fiscal"].get("horizon", 12)
    future_index = FI.fy_month_index(forecast_fy, start_month, horizon)
    for m in future_index:
        fm_of[m] = int(FI.fiscal_month(pd.Series([m.to_timestamp()]), start_month).iloc[0])
        fy_of[m] = forecast_fy

    # ----------------------------------------------------------- 5. profile
    status("Profiling every product...", 0.18)
    res.profile = FE.profile_products(panel, sales, cfg, last_fy=last_fy)

    # --------------------------------------------------------- 6. back-test
    n_models_done = {"n": 0}

    def prog(_name: str) -> None:
        n_models_done["n"] += 1
        frac = 0.20 + min(n_models_done["n"] / 60.0, 1.0) * 0.45
        status(f"Back-testing models ({n_models_done['n']} fitted)...", frac)

    status("Back-testing candidate models on past years...", 0.20)
    # Customer/order-line panels for the ML model's engineered features
    # (docs/IDEAS.md I). Built once; leakage-safe per origin by construction.
    from .global_ml import AuxPanels
    ml_aux = AuxPanels.build(sales, customers, items, panel.index,
                             list(panel.columns))
    detail, fold_store = BT.run_backtest(
        panel, items, cfg, fy_of, fm_of, n_jobs=n_jobs, progress=prog,
        aux=ml_aux)
    res.backtest_detail = detail

    status("Choosing the best model for each product...", 0.66)
    selection = BT.select_models(detail, cfg)

    # Products with no back-test result at all (e.g. brand new) still need a model.
    missing = panel.index.difference(selection["item_id"] if not selection.empty
                                     else pd.Index([]))
    if len(missing):
        prof = res.profile.reindex(missing)
        fallback = np.where(prof["tier"].to_numpy() == FE.TIER_DORMANT, "zero",
                            np.where(prof["tier"].to_numpy() == FE.TIER_SPARSE,
                                     "croston_sba", "moving_average_6"))
        extra = pd.DataFrame({
            "item_id": missing, "model": fallback,
            "tier": prof["tier"].to_numpy(),
            "backtest_WAPE": np.nan, "backtest_MASE": np.nan,
            "backtest_MAE": np.nan, "backtest_MAPE": np.nan,
            "folds_tested": 0,
            "selection_reason": "too little history to back-test - safe default used",
            "runner_up": "",
        })
        extra["model_plain"] = extra["model"].map(MOD.plain_name)
        selection = pd.concat([selection, extra], ignore_index=True)

    # The back-test assigned tiers from the training window only (to avoid
    # leakage). For reporting, use the tier based on the full history so the
    # dashboard and the forecast files agree with the product profile.
    selection["backtest_tier"] = selection["tier"]
    selection["tier"] = selection["item_id"].map(res.profile["tier"]).fillna(
        selection["tier"])

    # Dormant products are always forecast as zero, whatever back-tested best.
    dormant = res.profile.index[res.profile["tier"] == FE.TIER_DORMANT]
    m = selection["item_id"].isin(dormant)
    selection.loc[m, "model"] = "zero"
    selection.loc[m, "model_plain"] = MOD.plain_name("zero")
    selection.loc[m, "selection_reason"] = (
        f"no sales at all in FY{last_fy} - forecast as zero until the business "
        "confirms it is returning")
    # Ensemble policy annotation, so the per-product rationale stays truthful.
    if cfg["models"].get("ensemble_top_k", 1) >= 2 and "runner_up" in selection.columns:
        bl = ((selection["runner_up"].fillna("").astype(str) != "")
              & (selection["runner_up"] != "zero") & (selection["model"] != "zero"))
        selection.loc[bl, "selection_reason"] = (
            selection.loc[bl, "selection_reason"].astype(str)
            + " | Delivered forecast is the average of this method and the "
              "runner-up (ensemble policy, back-tested at -4.3% error)")

    res.selection = selection

    res.backtest_summary = BT.overall_accuracy(detail, selection)
    res.accuracy_delivered = BT.accuracy_as_delivered(
        fold_store, selection, panel, cfg, fm_of)
    res.benchmark = BT.benchmark_comparison(
        fold_store, selection, panel, cfg, fm_of)
    status("Building the inventory decision frontier...", 0.69)
    res.frontier = INV.build_frontier(fold_store, selection, panel, cfg, fm_of)
    res.accuracy_levels = BT.accuracy_by_level(
        fold_store, selection, panel, cfg, fm_of, items)

    # ------------------------------------------------ 7. reconciliation test
    status("Testing whether category reconciliation improves accuracy...", 0.70)
    if cfg["reconciliation"].get("test_on_backtest", True):
        res.reconciliation = FC.test_reconciliation(
            fold_store, selection, panel, items, cfg, fy_of, fm_of)
    else:
        res.reconciliation = {"enabled": cfg["reconciliation"].get("enabled", False),
                              "reason": "Reconciliation set manually in config."}

    # -------------------------------------------------------- 8. forecast FY
    status(f"Forecasting FY{forecast_fy}...", 0.74)
    long, group_fc, recon_audit = FC.build_forecast(
        panel, items, selection, cfg, fy_of, fm_of, future_index,
        reconciliation_decision=res.reconciliation, n_jobs=n_jobs, aux=ml_aux)
    res.group_forecast = group_fc

    # ------------------------------------------------------- 9. price & revenue
    status("Working out prices and revenue...", 0.86)
    prices = PR.resolve_unit_prices(sales, items, cfg, item_ids=panel.index)
    res.prices = prices
    rates = PR.billing_rate(sales, "item_id")
    default_rate = float(
        sales.loc[sales["is_revenue"], "quantity"].sum() / max(sales["quantity"].sum(), 1)
    )
    # VIP discount - see config.yaml pricing.vip_discount
    vip_mult, vip_summary = PR.vip_discount_adjustment(sales, customers, cfg)
    # Always compute BOTH the undiscounted and the 37%-discounted revenue so
    # the dashboard can show the other figure whichever setting is active.
    _sens_cfg = {"pricing": {**cfg.get("pricing", {}), "vip_discount": 0.37}}
    vip_sens_mult, _ = PR.vip_discount_adjustment(sales, customers, _sens_cfg)
    if vip_summary.get("applied"):
        prices = prices.copy()
        prices["unit_price"] = prices["unit_price"] * prices["item_id"].map(vip_mult).fillna(1.0)

    long = PR.apply_revenue(long, prices, rates, default_rate)

    # Validate the pricing half of the revenue calculation on the last known
    # year, so any price/mix bias is visible rather than buried in the total.
    revenue_check = PR.backtest_revenue_method(
        sales, prices, rates, default_rate, test_fy=last_fy)

    # Attach product attributes for filtering in the dashboard.
    if not items.empty:
        cols = [c for c in ("item_code", "category_1", "category_2",
                            "production_status", "list_price") if c in items.columns]
        long = long.merge(items[["item_id"] + cols], on="item_id", how="left")
    long["fiscal_year"] = forecast_fy
    long["month_label"] = long["month"].astype(str)
    res.forecast = long

    # Per-product annual summary.
    agg = long.groupby("item_id", observed=True).agg(
        forecast_quantity=("forecast_quantity", "sum"),
        forecast_revenue=("forecast_revenue", "sum"),
        unit_price=("unit_price", "first"),
        price_source=("price_source", "first"),
        revenue_flag=("revenue_flag", "first"),
        model=("model", "first"), model_plain=("model_plain", "first"),
        tier=("tier", "first"),
        backtest_WAPE=("backtest_WAPE", "first"),
        backtest_MAPE=("backtest_MAPE", "first"),
        backtest_MASE=("backtest_MASE", "first"),
    ).reset_index()
    if cfg["output"].get("round_annual_totals", True):
        agg["forecast_quantity"] = agg["forecast_quantity"].round(0)
        agg["forecast_revenue"] = agg["forecast_revenue"].round(0)
    if not items.empty:
        cols = [c for c in ("item_code", "category_1", "category_2",
                            "production_status") if c in items.columns]
        agg = agg.merge(items[["item_id"] + cols], on="item_id", how="left")
    prof_cols = ["active_months", "demand_pattern", "confidence", "last_sale_month"]
    agg = agg.merge(res.profile[prof_cols].reset_index(), on="item_id", how="left")

    # Event exposure: the size a single month can plausibly reach, from the
    # product's own record. Spike TIMING was shown to be statistically
    # unpredictable (docs/IDEAS.md A1-A3), so the honest planning aid is the
    # spike SIZE: the 90th percentile of the product's non-zero months (its
    # maximum where history is too short for a quantile). Used on the
    # Inventory Planning page to size event buffers for spike-prone products.
    vals = panel.to_numpy(dtype=float)
    expo = np.zeros(len(panel))
    for _i, _r in enumerate(vals):
        _nz = _r[_r > 0]
        expo[_i] = (np.quantile(_nz, 0.9) if len(_nz) >= 4
                    else (_nz.max() if len(_nz) else 0.0))
    agg["event_exposure_p90_month"] = agg["item_id"].map(
        dict(zip(panel.index, np.round(expo, 1)))).fillna(0.0)
    res.forecast_by_product = agg

    # --------------------------------------------- 9b. prediction intervals
    status("Working out how wide the uncertainty really is...", 0.90)
    res.interval_multipliers, interval_summary = IV.compute_multipliers(
        fold_store, selection, panel, cfg, fm_of, res.profile,
        backtest_detail=detail)
    long, agg = IV.apply_intervals(long, agg, res.interval_multipliers, res.profile)
    res.forecast = long
    res.forecast_by_product = agg

    # ------------------------------------------------- 10. history for the UI
    status("Preparing the history views...", 0.92)
    sales["revenue_billed"] = np.where(
        sales["is_revenue"], sales["quantity"] * sales["unit_price"].where(
            sales["unit_price"] > 0, np.nan), 0.0)

    res.history_item_month = (
        sales[sales["is_demand"]]
        .groupby(["item_id", "month"], observed=True)
        .agg(quantity=("quantity", "sum"), revenue=("revenue_billed", "sum"))
        .reset_index()
    )
    if "customer_id" in sales.columns:
        res.history_customer_month = (
            sales[sales["is_demand"]]
            .groupby(["customer_id", "month"], observed=True)
            .agg(quantity=("quantity", "sum"), revenue=("revenue_billed", "sum"))
            .reset_index()
        )
    res.history_total = (
        sales[sales["is_demand"]]
        .groupby("month", observed=True)
        .agg(quantity=("quantity", "sum"), revenue=("revenue_billed", "sum"))
        .reset_index()
    )
    res.history_total["fiscal_year"] = res.history_total["month"].map(fy_of)

    # ------------------------------- 11. allocate the forecast to customers
    # Product-level demand is forecast, then split across customers using each
    # customer's share of that product over the last 12 months. Customer-level
    # series are far too sparse to forecast directly; this is an allocation,
    # not a forecast, and is labelled as such in the dashboard.
    if "customer_id" in sales.columns:
        recent_cut = max(months) - 11
        recent = sales[(sales["month"] >= recent_cut) & sales["is_demand"]]
        share = (recent.groupby(["item_id", "customer_id"], observed=True)["quantity"]
                 .sum().reset_index())
        tot = share.groupby("item_id", observed=True)["quantity"].transform("sum")
        share["share"] = np.where(tot > 0, share["quantity"] / tot, 0.0)
        cust_fc = long[["item_id", "month", "forecast_quantity", "forecast_revenue"]].merge(
            share[["item_id", "customer_id", "share"]], on="item_id", how="inner")
        cust_fc["forecast_quantity"] *= cust_fc["share"]
        cust_fc["forecast_revenue"] *= cust_fc["share"]
        res.customer_forecast = cust_fc.drop(columns="share")

    # -------------------------------------------------- 12. accuracy by tier
    if not detail.empty and not selection.empty:
        picked = detail.merge(selection[["item_id", "model"]],
                              on=["item_id", "model"], how="inner")
        res.accuracy_by_tier = (
            picked.groupby(["tier", "fold"], observed=True)
            .apply(lambda g: pd.Series({
                "Products": g["item_id"].nunique(),
                "Actual units": g["actual_total"].sum(),
                "Forecast units": g["forecast_total"].sum(),
                "WAPE": (g["MAE"] * g["n_months"]).sum() / g["actual_total"].sum()
                if g["actual_total"].sum() > 0 else np.nan,
                "Median MAPE": g["MAPE"].median(),
                "Median MASE": g["MASE"].median(),
                "Median MAE": g["MAE"].median(),
            }), include_groups=False).reset_index()
        )

    # ------------------------------------------------------------ 13. meta
    _q_total = float(agg["forecast_quantity"].sum())
    _r_total = float(agg["forecast_revenue"].sum(skipna=True))
    # Revenue with the VIP discount backed out (== forecast_revenue when the
    # discount is switched off). Multiplier is >= 0.63 so division is safe.
    _applied_mult = (agg["item_id"].map(vip_mult).fillna(1.0)
                     if vip_summary.get("applied") else 1.0)
    _undiscounted_rev = agg["forecast_revenue"].fillna(0) / _applied_mult
    res.meta = {
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 1),
        "fiscal_start_month": start_month,
        "ensemble_top_k": cfg["models"].get("ensemble_top_k", 1),
        "fiscal_year_label": FI.MONTH_ABBR[start_month - 1] + "-" +
                             FI.MONTH_ABBR[(start_month - 2) % 12],
        "fiscal_detection_confidence": det.confidence,
        "fiscal_detection_evidence": det.evidence + ([peak_note] if peak_note else []),
        "fiscal_detection_matches_config": det.start_month == configured,
        "history_first_month": str(min(months)),
        "history_last_month": str(max(months)),
        "history_months": len(months),
        "history_fiscal_years": sorted({fy_of[m] for m in months}),
        "forecast_fiscal_year": forecast_fy,
        "forecast_label": FI.fy_label(forecast_fy, start_month),
        "forecast_months": [str(m) for m in future_index],
        "rows_in": qrep.rows_in,
        "rows_out": qrep.rows_out,
        "n_products": int(panel.shape[0]),
        "n_customers": int(sales["customer_id"].nunique()) if "customer_id" in sales else 0,
        "n_products_forecast_nonzero": int((agg["forecast_quantity"] > 0).sum()),
        "total_forecast_quantity": float(agg["forecast_quantity"].sum()),
        "total_forecast_revenue": float(agg["forecast_revenue"].sum(skipna=True)),
        "n_price_tbd": int((agg["revenue_flag"] == "PRICE_TBD").sum()),
        "n_price_variable": int((agg["revenue_flag"] == "PRICE_VARIABLE_ESTIMATED").sum()),
        "realised_to_list_ratio": prices.attrs.get("realised_to_list_ratio"),
        "ratio_source": prices.attrs.get("ratio_source"),
        "default_billing_rate": default_rate,
        "revenue_method_check": revenue_check,
        "vip_discount_question": vip_summary,
        "revenue_without_vip_discount": float(_undiscounted_rev.sum()),
        "revenue_if_vip_discount_37pct": float(
            (agg["item_id"].map(vip_sens_mult).fillna(1.0) * _undiscounted_rev).sum()),
        "intervals": interval_summary,
        # Company-wide range comes from portfolio-level evidence. Summing the
        # per-product ranges would assume every product has its worst year at
        # the same time, which produced a meaningless 74k-367k band.
        "forecast_quantity_low": _q_total * interval_summary.get(
            "portfolio_low_multiplier", float("nan")),
        "forecast_quantity_high": _q_total * interval_summary.get(
            "portfolio_high_multiplier", float("nan")),
        "forecast_revenue_low": _r_total * interval_summary.get(
            "portfolio_low_multiplier", float("nan")),
        "forecast_revenue_high": _r_total * interval_summary.get(
            "portfolio_high_multiplier", float("nan")),
        "forecast_quantity_low_if_summed": float(agg["forecast_quantity_low"].sum())
            if "forecast_quantity_low" in agg else None,
        "forecast_quantity_high_if_summed": float(agg["forecast_quantity_high"].sum())
            if "forecast_quantity_high" in agg else None,
        "reconciliation": res.reconciliation,
        "tier_counts": res.profile["tier"].value_counts().to_dict(),
        "model_counts": selection["model_plain"].value_counts().to_dict(),
        "config_used": cfg,
    }

    if write_outputs:
        status("Writing output files...", 0.97)
        write_all(res, recon_audit)

    status("Done.", 1.0)
    return res


def write_all(res: PipelineResult, recon_audit: pd.DataFrame | None = None) -> None:
    """Write every deliverable to outputs/."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fy = res.meta.get("forecast_fiscal_year", "next")

    # -------- the headline deliverable: product x month x quantity x revenue
    cols = [
        ("item_id", "Item Internal ID"),
        ("item_code", "Item Code"),
        ("category_1", "Product Category 1"),
        ("category_2", "Product Category 2"),
        ("production_status", "Production Status"),
        ("month_label", "Month"),
        ("fiscal_month", "Fiscal month (1=first month of FY)"),
        ("forecast_quantity", "Forecast quantity (units)"),
        ("forecast_quantity_low", "Forecast quantity - low estimate"),
        ("forecast_quantity_high", "Forecast quantity - high estimate"),
        ("unit_price", "Unit price used (THB)"),
        ("price_source", "Price basis"),
        ("billing_rate", "Historical billed share"),
        ("forecast_quantity_billable", "Forecast billable quantity"),
        ("forecast_revenue", "Forecast revenue (THB)"),
        ("forecast_revenue_low", "Forecast revenue - low estimate (THB)"),
        ("forecast_revenue_high", "Forecast revenue - high estimate (THB)"),
        ("revenue_flag", "Revenue flag"),
        ("model_plain", "Forecast method used"),
        ("tier", "Product segment"),
        ("backtest_WAPE", "Back-test WAPE"),
        ("backtest_MAPE", "Back-test MAPE %"),
        ("backtest_MASE", "Back-test MASE"),
    ]
    have = [(c, n) for c, n in cols if c in res.forecast.columns]
    out = res.forecast[[c for c, _ in have]].rename(columns=dict(have))
    out.to_csv(OUTPUTS / f"FY{fy}_forecast_by_product_month.csv", index=False)

    res.forecast_by_product.to_csv(
        OUTPUTS / f"FY{fy}_forecast_summary_by_product.csv", index=False)

    if not res.customer_forecast.empty:
        res.customer_forecast.assign(month=res.customer_forecast["month"].astype(str)) \
            .to_csv(OUTPUTS / f"FY{fy}_forecast_by_customer.csv", index=False)

    # ------------------------------------------------------ accuracy evidence
    if not res.backtest_summary.empty:
        res.backtest_summary.to_csv(OUTPUTS / "backtest_accuracy_summary.csv", index=False)
    if not res.accuracy_delivered.empty:
        res.accuracy_delivered.to_csv(
            OUTPUTS / "backtest_accuracy_as_delivered.csv", index=False)
    if not res.frontier.empty:
        res.frontier.to_csv(OUTPUTS / "inventory_service_frontier.csv", index=False)
    if not res.accuracy_levels.empty:
        res.accuracy_levels.to_csv(
            OUTPUTS / "accuracy_by_decision_level.csv", index=False)
    if not res.benchmark.empty:
        res.benchmark.drop(columns=[c for c in ("_ours",) if c in res.benchmark],
                           errors="ignore").to_csv(
            OUTPUTS / "benchmark_comparison.csv", index=False)
    if not res.accuracy_by_tier.empty:
        res.accuracy_by_tier.to_csv(OUTPUTS / "backtest_accuracy_by_segment.csv", index=False)
    if not res.backtest_detail.empty:
        picked = res.backtest_detail.merge(
            res.selection[["item_id", "model"]], on=["item_id", "model"], how="inner")
        picked.to_csv(OUTPUTS / "backtest_accuracy_by_product.csv", index=False)
        res.backtest_detail.to_csv(OUTPUTS / "backtest_all_models_scored.csv", index=False)

    res.selection.to_csv(OUTPUTS / "model_selection_by_product.csv", index=False)
    res.quality.to_csv(OUTPUTS / "data_quality_report.csv", index=False)

    prof = res.profile.copy()
    for c in ("first_sale_month", "last_sale_month"):
        if c in prof.columns:
            prof[c] = prof[c].astype(str)
    prof.reset_index().to_csv(OUTPUTS / "product_profile.csv", index=False)

    res.prices.to_csv(OUTPUTS / "unit_price_resolution.csv", index=False)
    if not res.interval_multipliers.empty:
        res.interval_multipliers.to_csv(
            OUTPUTS / "prediction_interval_multipliers.csv", index=False)
    if recon_audit is not None and not recon_audit.empty:
        recon_audit.to_csv(OUTPUTS / "reconciliation_audit.csv", index=False)

    with open(OUTPUTS / "reconciliation_decision.json", "w", encoding="utf-8") as fh:
        json.dump(res.reconciliation, fh, indent=2, default=str)

    with open(OUTPUTS / "run_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(res.meta, fh, indent=2, default=str)

    # -------- cached bundle so the dashboard loads instantly on the next visit
    bundle = {
        "forecast": res.forecast,
        "forecast_by_product": res.forecast_by_product,
        "customer_forecast": res.customer_forecast,
        "selection": res.selection,
        "backtest_detail": res.backtest_detail,
        "backtest_summary": res.backtest_summary,
        "accuracy_delivered": res.accuracy_delivered,
        "benchmark": res.benchmark,
        "frontier": res.frontier,
        "accuracy_levels": res.accuracy_levels,
        "frontier_test_fy": res.frontier.attrs.get("test_fy") if not res.frontier.empty else None,
        "frontier_total_demand": res.frontier.attrs.get("total_demand") if not res.frontier.empty else None,
        "interval_multipliers": res.interval_multipliers,
        "accuracy_by_tier": res.accuracy_by_tier,
        "profile": res.profile,
        "prices": res.prices,
        "quality": res.quality,
        "history_item_month": res.history_item_month,
        "history_customer_month": res.history_customer_month,
        "history_total": res.history_total,
        "items": res.items,
        "customers": res.customers,
        "group_forecast": res.group_forecast,
        "meta": res.meta,
    }
    pd.to_pickle(bundle, OUTPUTS / "dashboard_bundle.pkl")


def load_bundle(path: Path | None = None) -> dict | None:
    """Load the cached dashboard bundle, if a pipeline run has already happened."""
    p = Path(path) if path else OUTPUTS / "dashboard_bundle.pkl"
    if not p.exists():
        return None
    try:
        return pd.read_pickle(p)
    except Exception:
        return None
