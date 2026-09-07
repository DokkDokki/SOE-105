"""
clean_data.py
=============
CLEANING ONLY -- this script does not build features, aggregate, or shape
data for any model. It hands off clean, well-typed tables; the modeling
team decides what to do with them from there.

Why this is written the way it is
----------------------------------
This pipeline runs on a **new data extract every year**, and the exact rows
that are broken will differ each time. So cleaning happens in two layers:

  LAYER 1 -- EXPLICIT RULES (see COLUMN_RULES below)
      Every column we know about has a documented rule stating exactly
      what kind of problem to check for and what to do about it, based on
      what that column *is* (a primary key, a closed-list category, a
      required field, a free numeric measure, etc.) -- not based on which
      specific bad values happen to show up this year.

  LAYER 2 -- DEFAULT FALLBACK RULE (applies to everything NOT in Layer 1)
      Any column with no explicit rule -- because it's new this year, or
      because an error type shows up in a column no one anticipated -- is
      still cleaned automatically:
          - if more than 10% of that column's values are bad -> AVERAGE
            (mean for numeric columns, most-common value for text columns)
          - if 10% or fewer are bad -> DELETE just those rows
      This guarantees nothing falls through uncleaned, without needing the
      code to be updated every year.

What counts as "bad" for the fallback rule: a null value, a value that
failed to parse as its expected type, or (for numeric columns) a value
below zero where the business rule says the column can't be negative.

Explicit rule types (Layer 1)
------------------------------
  key             - a primary key column: rows with a null key are dropped
                     (can't average or guess an ID), then the table is
                     de-duplicated on that key.
  foreign_key     - a key referencing another table: rows with a null key
                     or a key that doesn't exist in the referenced table's
                     primary key are dropped (an ID can't be averaged).
  required        - must never be null; null rows are dropped (there's no
                     sensible "average" for e.g. a company name).
  enum            - a closed-list categorical field. Values outside the
                     documented allowed set are FLAGGED in a
                     `<column>_out_of_domain` column, never deleted or
                     changed -- an unfamiliar category is still real
                     business data, just worth a human's attention.
  optional_text   - free-text/optional category that can legitimately be
                     blank; missing values are filled with "Unspecified"
                     rather than dropped or averaged.
  numeric_measure - numeric, optional, has a business floor (usually >= 0).
                     Uses the same 10%-threshold logic as the fallback rule,
                     applied deliberately because we know in advance this
                     column will have this kind of problem every year
                     (missing / out-of-range prices, quantities, etc.).

Row-level checks (apply across a whole row, not one column)
-------------------------------------------------------------
  - exact duplicate rows -> flagged (`is_duplicate_row`), not deleted,
    because a repeated line item can be a legitimate second order line.

Usage
-----
    python clean_data.py --input-dir /path/to/raw --output-dir /path/to/clean

Outputs
-------
    customer_clean.csv, item_clean.csv, sales_clean.csv, cleaning_report.json

Reusable functions (importable elsewhere):
    load_raw, clean_customer, clean_item, clean_sales
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Threshold for the fallback rule: > this fraction bad -> average/impute,
# <= this fraction bad -> delete the affected rows.
# --------------------------------------------------------------------------- #
BAD_DATA_THRESHOLD = 0.10

NULL_LIKE_STRINGS = {
    "", "-", "--", "n/a", "na", "none", "- none -", "null", "nan",
    "unknown", "not specified", "not available", "#n/a", "#null!",
}

# --------------------------------------------------------------------------- #
# LAYER 1 -- explicit, documented rule for every column we know about.
# This table IS the "what row/column needs what cleaning" specification.
# Anything not listed here automatically falls through to Layer 2.
# --------------------------------------------------------------------------- #
COLUMN_RULES: Dict[str, Dict[str, dict]] = {
    "customer": {
        "Customer Internal ID": {"kind": "key"},
        "Company Name":         {"kind": "required"},
        "Customer Type":        {"kind": "enum", "allowed": {"VIP", "End User"}},
        "FA Sub Industries":    {"kind": "optional_text"},
    },
    "item": {
        "Item Internal ID":   {"kind": "key"},
        "Item Code":          {"kind": "required"},
        "Production Status":  {"kind": "enum", "allowed": {"Continue", "Discontinue"}},
        "Product Category 1": {"kind": "enum", "allowed": None},  # open list -> flag & surface, don't restrict
        "Product Category 2": {"kind": "optional_text"},
        "List price":         {"kind": "numeric_measure", "min": 0},
    },
    "sales": {
        "SO Internal ID":       {"kind": "required"},  # line id; can repeat legitimately, so not deduped as a key
        "Sales Order Date":     {"kind": "required_date"},
        "Customer Internal ID": {"kind": "foreign_key", "references": "customer"},
        "Item Internal ID":     {"kind": "foreign_key", "references": "item"},
        "Sales Order Status":   {"kind": "enum", "allowed": {
            "Billed", "Closed", "Partially Fulfilled", "Pending Billing",
            "Pending Fulfillment", "Pending Billing/Partially Fulfilled",
        }},
        "Quantity":   {"kind": "numeric_measure", "min": 0},
        "Base Price": {"kind": "numeric_measure", "min": 0},
    },
}


# --------------------------------------------------------------------------- #
# Small generic helpers
# --------------------------------------------------------------------------- #

def _log_action(report: dict, key: str, column: str, info: dict) -> None:
    report.setdefault(f"{key}_actions", {})[column] = info


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def normalize_null_like(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Any placeholder string that really means missing becomes real NaN,
    for a single column -- applied selectively (see kind dispatch below),
    never to closed-list enum columns, where an unfamiliar string is a
    real category to flag rather than a null to erase."""
    df = df.copy()
    if column not in df.columns:
        return df
    as_text = df[column].astype("string").str.strip().str.lower()
    mask = as_text.isin(NULL_LIKE_STRINGS)
    df.loc[mask, column] = pd.NA
    return df


def try_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# --------------------------------------------------------------------------- #
# LAYER 1 -- dispatch: apply the documented rule for one column
# --------------------------------------------------------------------------- #

def apply_key_rule(df: pd.DataFrame, column: str, key: str, report: dict) -> pd.DataFrame:
    df = df.copy()
    df[column] = try_numeric(df[column])
    n_null = int(df[column].isna().sum())
    if n_null:
        df = df.dropna(subset=[column])
    before = len(df)
    df = df.drop_duplicates(subset=[column], keep="first")
    n_dup = before - len(df)
    _log_action(report, key, column, {
        "rule": "key", "null_rows_dropped": n_null, "duplicate_key_rows_dropped": n_dup,
    })
    return df


def apply_foreign_key_rule(df: pd.DataFrame, column: str, ref_df: pd.DataFrame, ref_key_col: str, key: str, report: dict) -> pd.DataFrame:
    df = df.copy()
    df[column] = try_numeric(df[column])
    n_null = int(df[column].isna().sum())
    df = df.dropna(subset=[column])
    orphan_mask = ~df[column].isin(ref_df[ref_key_col])
    n_orphan = int(orphan_mask.sum())
    df = df[~orphan_mask]
    _log_action(report, key, column, {
        "rule": "foreign_key", "null_rows_dropped": n_null, "orphan_rows_dropped": n_orphan,
    })
    return df


def apply_required_rule(df: pd.DataFrame, column: str, key: str, report: dict) -> pd.DataFrame:
    df = normalize_null_like(df, column)
    n_null = int(df[column].isna().sum())
    if n_null:
        df = df.dropna(subset=[column])
    _log_action(report, key, column, {"rule": "required", "null_rows_dropped": n_null})
    return df


def apply_required_date_rule(df: pd.DataFrame, column: str, key: str, report: dict) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    n_null = int(df[column].isna().sum())
    if n_null:
        df = df.dropna(subset=[column])
    _log_action(report, key, column, {"rule": "required_date", "unparseable_rows_dropped": n_null})
    return df


def apply_enum_rule(df: pd.DataFrame, column: str, allowed: Optional[set], key: str, report: dict) -> pd.DataFrame:
    df = df.copy()
    flag_col = f"{column}_out_of_domain"
    distribution = df[column].value_counts(dropna=True).to_dict()
    if allowed is not None:
        df[flag_col] = df[column].notna() & ~df[column].isin(allowed)
        n_flagged = int(df[flag_col].sum())
        unexpected = sorted(set(df.loc[df[flag_col], column].dropna().unique())) if n_flagged else []
    else:
        df[flag_col] = False
        n_flagged = 0
        unexpected = []
    _log_action(report, key, column, {
        "rule": "enum", "flagged_out_of_domain": n_flagged,
        "unexpected_values": unexpected, "value_distribution": distribution,
    })
    return df


def apply_optional_text_rule(df: pd.DataFrame, column: str, key: str, report: dict) -> pd.DataFrame:
    df = normalize_null_like(df, column)
    n_null = int(df[column].isna().sum())
    df[column] = df[column].fillna("Unspecified")
    _log_action(report, key, column, {"rule": "optional_text", "filled_unspecified": n_null})
    return df


def apply_numeric_measure_rule(df: pd.DataFrame, column: str, min_value: Optional[float], key: str, report: dict) -> pd.DataFrame:
    """Deliberate, explicit use of the same delete-vs-average threshold
    logic as the Layer-2 fallback -- used here on purpose because we know
    numeric measures like price/quantity reliably need this treatment
    every year, not because the column is unclassified."""
    df = df.copy()
    df[column] = try_numeric(df[column])
    error_mask = df[column].isna()
    if min_value is not None:
        error_mask = error_mask | (df[column] < min_value)
    df = _resolve_by_threshold(df, column, error_mask, key, report, rule_name="numeric_measure")
    return df


# --------------------------------------------------------------------------- #
# LAYER 2 -- fallback rule for any column with no entry in COLUMN_RULES
# --------------------------------------------------------------------------- #

def _resolve_by_threshold(
    df: pd.DataFrame, column: str, error_mask: pd.Series, key: str, report: dict, rule_name: str = "default_fallback",
) -> pd.DataFrame:
    df = df.copy()
    n_total = len(df)
    n_err = int(error_mask.sum())
    if n_err == 0:
        _log_action(report, key, column, {"rule": rule_name, "error_count": 0, "action": "none needed"})
        return df

    error_rate = n_err / n_total
    is_numeric = pd.api.types.is_numeric_dtype(df[column])

    if error_rate > BAD_DATA_THRESHOLD:
        # More than 10% bad -> average it in (mean for numeric, most
        # common value for text) rather than losing that much of the table.
        valid = df.loc[~error_mask, column]
        if is_numeric:
            fill_value = float(valid.mean()) if len(valid) else 0.0
            action = f"averaged (mean = {fill_value:.4g})"
        else:
            mode = valid.mode()
            fill_value = mode.iloc[0] if not mode.empty else "Unspecified"
            action = f"imputed with most common value ({fill_value!r})"
        df.loc[error_mask, column] = fill_value
    else:
        # 10% or fewer bad -> just drop those rows.
        df = df[~error_mask]
        action = "rows dropped"

    _log_action(report, key, column, {
        "rule": rule_name, "error_count": n_err,
        "error_rate_pct": round(error_rate * 100, 2), "action": action,
    })
    return df


def apply_default_fallback(df: pd.DataFrame, column: str, key: str, report: dict) -> pd.DataFrame:
    """Layer 2: this column has no entry in COLUMN_RULES, so we don't know
    its business meaning. Treat any null or unparseable value as an error
    and resolve it with the 10% delete-vs-average rule."""
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[column]) or df[column].dtype == object:
        # Try numeric coercion first; if that destroys almost everything,
        # it wasn't really numeric, so fall back to treating it as text.
        coerced = try_numeric(df[column])
        if coerced.notna().sum() >= 0.5 * df[column].notna().sum():
            df[column] = coerced
    error_mask = df[column].isna()
    return _resolve_by_threshold(df, column, error_mask, key, report, rule_name="default_fallback (unmatched column)")


def flag_exact_duplicate_rows(df: pd.DataFrame, key: str, report: dict) -> pd.DataFrame:
    df = df.copy()
    dup_mask = df.duplicated(keep=False)
    df["is_duplicate_row"] = dup_mask
    report[f"{key}_exact_duplicate_rows_flagged"] = int(dup_mask.sum())
    return df


# --------------------------------------------------------------------------- #
# Table-level orchestration: apply Layer 1 to every ruled column, then
# Layer 2 to every remaining column automatically.
# --------------------------------------------------------------------------- #

def _clean_table(
    df: pd.DataFrame, table_key: str, report: dict, foreign_refs: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    df = clean_column_names(df)
    df = strip_string_values(df)
    rules = COLUMN_RULES.get(table_key, {})

    for column, rule in rules.items():
        if column not in df.columns:
            continue
        kind = rule["kind"]
        if kind == "key":
            df = apply_key_rule(df, column, table_key, report)
        elif kind == "foreign_key":
            ref_table = rule["references"]
            ref_df = foreign_refs[ref_table]
            ref_key_col = next(iter(COLUMN_RULES[ref_table]))  # first rule = the ref table's primary key
            df = apply_foreign_key_rule(df, column, ref_df, ref_key_col, table_key, report)
        elif kind == "required":
            df = apply_required_rule(df, column, table_key, report)
        elif kind == "required_date":
            df = apply_required_date_rule(df, column, table_key, report)
        elif kind == "enum":
            df = apply_enum_rule(df, column, rule.get("allowed"), table_key, report)
        elif kind == "optional_text":
            df = apply_optional_text_rule(df, column, table_key, report)
        elif kind == "numeric_measure":
            df = apply_numeric_measure_rule(df, column, rule.get("min"), table_key, report)
        else:
            raise ValueError(f"Unknown rule kind: {kind}")

    # LAYER 2: any column not covered by an explicit rule above.
    unmatched_cols = [c for c in df.columns if c not in rules]
    for column in unmatched_cols:
        df = apply_default_fallback(df, column, table_key, report)

    df = flag_exact_duplicate_rows(df, table_key, report)
    report[f"{table_key}_rows_clean"] = len(df)
    log.info("Cleaned %s: %s rows", table_key, len(df))
    return df


def clean_customer(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    return _clean_table(df, "customer", report)


def clean_item(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    return _clean_table(df, "item", report)


def clean_sales(df: pd.DataFrame, customer: pd.DataFrame, item: pd.DataFrame, report: dict) -> pd.DataFrame:
    return _clean_table(df, "sales", report, foreign_refs={"customer": customer, "item": item})


# --------------------------------------------------------------------------- #
# Loading + orchestration
# --------------------------------------------------------------------------- #

def load_raw(input_dir: Path) -> Dict[str, pd.DataFrame]:
    customer = pd.read_excel(input_dir / "Customer.xlsx")
    item = pd.read_excel(input_dir / "Item.xlsx")
    sales = pd.read_excel(input_dir / "Sale_Order_History.xlsx")
    log.info("Loaded raw shapes -> customer=%s item=%s sales=%s", customer.shape, item.shape, sales.shape)
    return {"customer": customer, "item": item, "sales": sales}


def run_pipeline(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    raw = load_raw(input_dir)
    customer = clean_customer(raw["customer"], report)
    item = clean_item(raw["item"], report)
    # sales cleaning needs the *cleaned* customer/item tables for the
    # foreign-key check, so it runs last.
    sales = clean_sales(raw["sales"], customer, item, report)

    customer.to_csv(output_dir / "customer_clean.csv", index=False)
    item.to_csv(output_dir / "item_clean.csv", index=False)
    sales.to_csv(output_dir / "sales_clean.csv", index=False)

    with open(output_dir / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("Done. Outputs written to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/mnt/user-data/uploads"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/user-data/outputs"))
    args = parser.parse_args()
    run_pipeline(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
