"""Command-line entry point.

    python run_pipeline.py                    # read from data/raw, write to outputs/
    python run_pipeline.py --data path/to/dir
    python run_pipeline.py --jobs 4           # limit CPU use
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import DATA_RAW, OUTPUTS
from src.pipeline import run_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Forecast next fiscal year demand and revenue.")
    ap.add_argument("--data", type=Path, default=None,
                    help="Folder containing the sales, item and customer files "
                         "(default: data/raw)")
    ap.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    ap.add_argument("--jobs", type=int, default=-1,
                    help="Parallel workers (-1 = all cores)")
    ap.add_argument("--quick", action="store_true",
                    help="Quick refresh: reuse model selection and accuracy "
                         "evidence from the last full run; only refit and "
                         "re-forecast (minutes instead of ~20)")
    args = ap.parse_args()

    def status(msg: str, frac: float) -> None:
        bar = "#" * int(frac * 30)
        print(f"\r[{bar:<30}] {frac*100:5.1f}%  {msg[:60]:<60}", end="", flush=True)

    try:
        res = run_pipeline(data_folder=args.data or DATA_RAW, config_path=args.config,
                           n_jobs=args.jobs, on_status=status, quick=args.quick)
    except Exception as exc:
        print(f"\n\nERROR: {exc}\n", file=sys.stderr)
        return 1

    m = res.meta
    print("\n")
    print("=" * 72)
    print(f"  {m['forecast_label']} forecast complete")
    print("=" * 72)
    print(f"  Fiscal year detected     : {m['fiscal_year_label']} "
          f"(confidence: {m['fiscal_detection_confidence']})")
    print(f"  History used             : {m['history_first_month']} to "
          f"{m['history_last_month']} ({m['history_months']} months)")
    print(f"  Order lines after cleaning: {m['rows_out']:,} of {m['rows_in']:,}")
    print(f"  Products                 : {m['n_products']:,} "
          f"({m['n_products_forecast_nonzero']:,} with a non-zero forecast)")
    print(f"  Forecast quantity        : {m['total_forecast_quantity']:,.0f} units")
    print(f"  Forecast revenue         : THB {m['total_forecast_revenue']:,.0f}")
    print(f"  Products priced TBD      : {m['n_price_tbd']:,}")
    print(f"  Runtime                  : {m['runtime_seconds']}s")
    print()
    if not res.backtest_summary.empty:
        print("  Back-test accuracy (selected models, raw):")
        print(res.backtest_summary.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    if not res.accuracy_delivered.empty:
        print()
        print("  Back-test accuracy AS DELIVERED (models + safeguards):")
        cols = ["Fold", "Test year", "Actual units", "Forecast units (as delivered)",
                "Total error % (as delivered)", "WAPE (as delivered)"]
        print(res.accuracy_delivered[cols].to_string(
            index=False, float_format=lambda v: f"{v:,.3f}"))
    print()
    print(f"  Outputs written to: {OUTPUTS}")
    print(f"  Start the dashboard with:  streamlit run app/dashboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
