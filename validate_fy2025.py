"""Evaluate frozen FY2025 forecasts against TNIC actuals; never train models.

Run: python validate_fy2025.py --actuals "C:/Users/Doki/Downloads/TNIC Data_2025.xlsx"
Outputs are kept separately under outputs/fy2025_validation/.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MONTHS = pd.period_range("2025-04", "2026-03", freq="M").astype(str)
FROZEN_FORECAST_COMMIT = "523dc6300ee9a722bbff0e2f3c5d41de879bc950"


def dates(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="D", origin="1899-12-30")
    return pd.to_datetime(series, errors="raise")


def metrics(actual, forecast):
    a, p = np.asarray(actual, float).ravel(), np.asarray(forecast, float).ravel()
    e = p - a
    nz = a != 0
    return {
        "actual_units": float(a.sum()), "forecast_units": float(p.sum()),
        "bias_units": float(e.sum()), "bias_pct": float(e.sum()/a.sum()*100) if a.sum() else np.nan,
        "WAPE_pct": float(np.abs(e).sum()/np.abs(a).sum()*100) if np.abs(a).sum() else np.nan,
        "RMSE_units": float(np.sqrt(np.mean(e**2))),
        "MAPE_pct_nonzero_actuals": float(np.mean(np.abs(e[nz]/a[nz]))*100) if nz.any() else np.nan,
        "MAPE_zero_cells_excluded": int((~nz).sum()), "cells": int(len(a)),
    }


def forecast_panel(frame, ids):
    assert not frame.duplicated(["Item Internal ID", "Month"]).any()
    assert set(frame.Month) == set(MONTHS)
    p = frame.pivot(index="Item Internal ID", columns="Month", values="Forecast quantity (units)")
    assert not p.isna().any().any()
    return p.reindex(index=ids, columns=MONTHS, fill_value=0).fillna(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actuals", type=Path, required=True)
    args = ap.parse_args()
    out = ROOT / "outputs/fy2025_validation"
    out.mkdir(parents=True, exist_ok=True)
    sales = pd.read_excel(args.actuals, sheet_name="FY25 - Sale Order History")
    master = pd.read_excel(args.actuals, sheet_name="Item")
    sales.columns = sales.columns.str.strip()
    master.columns = master.columns.str.strip()
    sales["date"] = dates(sales["Sales Order Date"])
    assert sales["date"].min() == pd.Timestamp("2025-04-01")
    assert sales["date"].max() == pd.Timestamp("2026-03-31")
    assert sales[["Item Internal ID", "Quantity"]].notna().all().all()
    assert (sales.Quantity >= 0).all()
    sales["Month"] = sales.date.dt.to_period("M").astype(str)
    rel = "outputs/FY2025_forecast_by_product_month.csv"
    local = pd.read_csv(ROOT / rel)
    commit = FROZEN_FORECAST_COMMIT
    git_bytes = subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=ROOT)
    original = pd.read_csv(io.BytesIO(git_bytes))
    # Preserve an immutable copy of both evaluated predictions.
    (out / "repository_forecast_snapshot.csv").write_bytes(git_bytes)
    (out / "local_forecast_snapshot.csv").write_bytes((ROOT / rel).read_bytes())
    known_ids = set(local["Item Internal ID"])
    assert known_ids == set(original["Item Internal ID"])
    code_map = local[["Item Internal ID", "Item Code"]].drop_duplicates()
    assert not code_map["Item Internal ID"].duplicated().any()
    mapped = sales["Item Internal ID"].map(code_map.set_index("Item Internal ID")["Item Code"])
    mismatches = mapped.notna() & mapped.ne(sales["Item Code"])
    assert not mismatches.any(), "Product IDs and codes disagree; cannot score safely."
    ids = sorted(known_ids | set(sales["Item Internal ID"]))
    actual = sales.pivot_table(index="Item Internal ID", columns="Month", values="Quantity", aggfunc="sum").reindex(index=ids, columns=MONTHS).fillna(0)
    predictions = {"Repository frozen forecast": forecast_panel(original, ids),
                   "Local regenerated forecast": forecast_panel(local, ids)}
    # Baselines use only the original training history through March 2025.
    history = pd.read_excel(ROOT / "data/raw/Sale Order History.xlsx")
    history.columns = history.columns.str.strip()
    history["date"] = dates(history["Sales Order Date"])
    assert history.date.max() < sales.date.min()
    history = history.loc[history.date.between("2024-04-01", "2025-03-31")]
    history["Month"] = (history.date.dt.to_period("M") + 12).astype(str)
    repeat = history.pivot_table(index="Item Internal ID", columns="Month", values="Quantity", aggfunc="sum").reindex(index=ids, columns=MONTHS).fillna(0)
    predictions["Repeat FY2024 by month"] = repeat
    predictions["FY2024 annual average"] = pd.DataFrame(np.repeat((repeat.sum(axis=1)/12).to_numpy()[:,None],12,axis=1), index=ids, columns=MONTHS)
    records = []
    for name, pred in predictions.items():
        for level, a, p in [("Product-month",actual,pred), ("Company-month",actual.sum(axis=0),pred.sum(axis=0)), ("Product-year",actual.sum(axis=1),pred.sum(axis=1))]:
            records.append({"forecast":name,"level":level,**metrics(a,p)})
    scores = pd.DataFrame(records)
    scores.to_csv(out / "accuracy_summary.csv", index=False)
    cat_map = master.drop_duplicates("Item Internal ID").set_index("Item Internal ID")["Product Category 1"]
    fallback = local.drop_duplicates("Item Internal ID").set_index("Item Internal ID")["Product Category 1"]
    categories = pd.Series(ids,index=ids).map(cat_map).fillna(pd.Series(ids,index=ids).map(fallback)).fillna("Unknown")
    detail = actual.stack().rename("actual_units").to_frame()
    for name, pred in list(predictions.items())[:2]:
        key = "repository" if name.startswith("Repository") else "local"
        detail[f"{key}_forecast_units"] = pred.stack()
        detail[f"{key}_error_units"] = detail[f"{key}_forecast_units"] - detail.actual_units
    detail.index.names = ["item_id","month"]
    detail = detail.reset_index()
    detail["category"] = detail.item_id.map(categories)
    detail["product_code"] = detail.item_id.map(master.drop_duplicates("Item Internal ID").set_index("Item Internal ID")["Item Code"]).fillna(detail.item_id.map(code_map.set_index("Item Internal ID")["Item Code"]))
    detail["outside_forecast_catalogue"] = ~detail.item_id.isin(known_ids)
    detail.to_csv(out / "product_month_comparison.csv",index=False)
    monthly = detail.groupby("month")[["actual_units","repository_forecast_units","local_forecast_units"]].sum()
    monthly["repository_bias_pct"] = (monthly.repository_forecast_units/monthly.actual_units-1)*100
    monthly.to_csv(out / "monthly_comparison.csv")
    products = detail.groupby(["item_id","product_code","category","outside_forecast_catalogue"],dropna=False)[["actual_units","repository_forecast_units","local_forecast_units"]].sum().reset_index()
    products["repository_error_units"] = products.repository_forecast_units-products.actual_units
    products["absolute_annual_error_units"] = products.repository_error_units.abs()
    products.sort_values("absolute_annual_error_units",ascending=False).to_csv(out / "product_year_comparison.csv",index=False)
    cat_rows = []
    for cat in sorted(categories.unique()):
        ix = categories.index[categories.eq(cat)]
        cat_rows.append({"category":cat,**metrics(actual.loc[ix],predictions["Repository frozen forecast"].loc[ix])})
    pd.DataFrame(cat_rows).sort_values("actual_units",ascending=False).to_csv(out / "category_accuracy.csv",index=False)
    new = sales.loc[~sales["Item Internal ID"].isin(known_ids)]
    p0 = predictions["Repository frozen forecast"]
    zero_annual = p0.sum(axis=1).eq(0)
    audit = {"actuals_file":str(args.actuals.resolve()), "actuals_sha256":hashlib.sha256(args.actuals.read_bytes()).hexdigest(),
             "repository_commit":commit,"repository_forecast_sha256":hashlib.sha256(git_bytes).hexdigest(),
             "local_forecast_sha256":hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),
             "actual_rows":len(sales),"date_start":str(sales.date.min().date()),"date_end":str(sales.date.max().date()),
             "duplicate_rows_kept":int(sales.drop(columns=["date","Month"]).duplicated().sum()),
             "product_code_mismatches":int(mismatches.sum()),"scored_products":len(ids),"forecast_catalogue_products":len(known_ids),
             "actual_active_products":sales["Item Internal ID"].nunique(),"outside_catalogue_products":new["Item Internal ID"].nunique(),
             "outside_catalogue_units":float(new.Quantity.sum()),"zero_annual_forecast_actual_units":float(actual.loc[zero_annual].to_numpy().sum()),
             "price_zero_rows":int(sales["Base Price"].eq(0).sum()),
             "notes":["Demand includes all order statuses, consistent with config.yaml.","Exact duplicate lines retained, consistent with the established ERP-line policy.","Union of forecast and actual product IDs; absent product-month lines count as zero. Products outside the forecast catalogue are scored as zero forecast and reported separately.","MAPE excludes zero-actual cells, with excluded counts reported. WAPE includes these cells.","No retraining or recalibration uses FY2025 actuals.","Revenue accuracy is not scored: actual invoice revenue is absent and zero Base Price means unresolved price; VIP discount and billing definitions require reconciliation.","Repository forecast is the committed pre-actuals result. Local forecast was regenerated during setup and is a separately labelled sensitivity check."]}
    (out/"validation_metadata.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(scores.to_string(index=False))
    print(json.dumps(audit,indent=2))
    print("\nMonthly comparison:\n", monthly.to_string())
    print("\nLargest annual misses:\n", products.nlargest(10,"absolute_annual_error_units").to_string(index=False))


if __name__ == "__main__":
    main()
