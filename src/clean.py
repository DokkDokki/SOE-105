"""Cleaning, reconciliation against the master lists, and the data-quality report.

Nothing here silently discards data. Every rule records what it changed and why,
so the team can show Mitsubishi exactly how the numbers were produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .fiscal import add_fiscal_columns


@dataclass
class QualityIssue:
    check: str
    severity: str          # "info" | "warning" | "error"
    count: int
    detail: str
    action: str


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    issues: list[QualityIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, check, severity, count, detail, action) -> None:
        self.issues.append(QualityIssue(check, severity, int(count), detail, action))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Check": i.check, "Severity": i.severity, "Rows affected": i.count,
            "What we found": i.detail, "What we did": i.action,
        } for i in self.issues])


def clean_sales(
    sales: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the sales history and reconcile it against the master lists."""
    rep = CleaningReport(rows_in=len(sales))
    df = sales.copy()
    cl = cfg["cleaning"]

    # ---------------------------------------------------------------- dates
    n_bad_date = int(df["order_date"].isna().sum())
    if n_bad_date:
        df = df[df["order_date"].notna()]
        rep.add("Unreadable order date", "error", n_bad_date,
                "Order date could not be parsed.",
                "Rows dropped - a transaction without a date cannot be placed in a month.")

    # ------------------------------------------------------------- quantity
    n_null_qty = int(df["quantity"].isna().sum())
    if n_null_qty:
        df = df[df["quantity"].notna()]
        rep.add("Missing quantity", "error", n_null_qty,
                "Quantity was blank or non-numeric.",
                "Rows dropped - quantity is the forecast target.")

    n_neg = int((df["quantity"] < 0).sum())
    if n_neg:
        rep.add("Negative quantity", "warning", n_neg,
                "Quantity below zero, usually a return or correction.",
                "Kept - they net off genuine demand in the monthly totals.")

    n_zero_qty = int((df["quantity"] == 0).sum())
    if n_zero_qty:
        if cl["drop_nonpositive_quantity"]:
            df = df[df["quantity"] != 0]
            rep.add("Zero quantity", "info", n_zero_qty,
                    "Order lines with quantity 0 - not real transactions.",
                    "Rows dropped (zero effect on demand and revenue).")
        else:
            rep.add("Zero quantity", "info", n_zero_qty,
                    "Order lines with quantity 0.", "Kept per configuration.")

    # ----------------------------------------------------------- duplicates
    dup_mask = df.duplicated(keep=False)
    n_dup_rows = int(df.duplicated().sum())
    if n_dup_rows:
        dup_qty_share = float(df.loc[df.duplicated(), "quantity"].sum() /
                              max(df["quantity"].sum(), 1))
        if cl["drop_exact_duplicates"]:
            df = df.drop_duplicates()
            rep.add("Exact duplicate order lines", "warning", n_dup_rows,
                    f"Identical rows carrying {dup_qty_share:.2%} of total quantity.",
                    "Dropped, as configured (drop_exact_duplicates: true).")
        else:
            rep.add("Exact duplicate order lines", "info", n_dup_rows,
                    f"Identical rows carrying {dup_qty_share:.2%} of total quantity. "
                    "Forensic checks indicate these are genuine repeated ERP line "
                    "items (only part of each order repeats; spread evenly across "
                    "all four years; 65% are qty=1).",
                    "KEPT - dropping them would understate real demand. "
                    "Set drop_exact_duplicates: true in config.yaml to reverse.")
        rep.stats["duplicate_rows"] = n_dup_rows
        rep.stats["duplicate_qty_share"] = dup_qty_share
    _ = dup_mask

    # ----------------------------------------------- master list reconciliation
    if not items.empty and "item_id" in items.columns:
        n_item_dupe = int(items["item_id"].duplicated().sum())
        if n_item_dupe:
            items = items.drop_duplicates(subset="item_id", keep="first")
            rep.add("Duplicate items in master list", "warning", n_item_dupe,
                    "The same Item Internal ID appears more than once.",
                    "First occurrence kept.")
        known_items = set(items["item_id"].dropna())
        orphan_mask = ~df["item_id"].isin(known_items)
        n_orphan_items = int(orphan_mask.sum())
        if n_orphan_items:
            rep.add("Sales for unknown products", "warning", n_orphan_items,
                    f"{df.loc[orphan_mask, 'item_id'].nunique()} product IDs are in the "
                    "sales history but not in the item master list.",
                    "Kept and forecast, but labelled 'Not in item master' so the "
                    "team can chase the missing master records.")
        else:
            rep.add("Sales for unknown products", "info", 0,
                    "Every product in the sales history exists in the item master list.",
                    "No action needed.")

    if not customers.empty and "customer_id" in customers.columns:
        n_cust_dupe = int(customers["customer_id"].duplicated().sum())
        if n_cust_dupe:
            customers = customers.drop_duplicates(subset="customer_id", keep="first")
            rep.add("Duplicate customers in master list", "warning", n_cust_dupe,
                    "The same Customer Internal ID appears more than once.",
                    "First occurrence kept.")
        known_cust = set(customers["customer_id"].dropna())
        if "customer_id" in df.columns:
            n_orphan_cust = int((~df["customer_id"].isin(known_cust)).sum())
            if n_orphan_cust:
                rep.add("Sales for unknown customers", "warning", n_orphan_cust,
                        "Customer IDs present in sales but missing from the customer list.",
                        "Kept and labelled 'Not in customer master'.")
            else:
                rep.add("Sales for unknown customers", "info", 0,
                        "Every customer in the sales history exists in the customer list.",
                        "No action needed.")

    # ------------------------------------- inconsistent naming (ID <-> label)
    if "item_code" in df.columns:
        multi = df.groupby("item_id")["item_code"].nunique(dropna=True)
        n_multi = int((multi > 1).sum())
        rep.add("Product ID with several product codes", "warning" if n_multi else "info",
                n_multi,
                f"{n_multi} product IDs carry more than one product code."
                if n_multi else "Product ID and product code agree one-to-one.",
                "The master list code is treated as authoritative."
                if n_multi else "No action needed.")

    if "customer_name" in df.columns:
        multi_c = df.groupby("customer_id")["customer_name"].nunique(dropna=True)
        n_multi_c = int((multi_c > 1).sum())
        rep.add("Customer ID with several company names", "warning" if n_multi_c else "info",
                n_multi_c,
                f"{n_multi_c} customer IDs carry more than one company name."
                if n_multi_c else "Customer ID and company name agree one-to-one.",
                "The master list name is treated as authoritative."
                if n_multi_c else "No action needed.")

    # ------------------------------------------------------------ unit price
    if "unit_price" in df.columns:
        n_null_price = int(df["unit_price"].isna().sum())
        if n_null_price:
            df["unit_price"] = df["unit_price"].fillna(0.0)
            rep.add("Missing unit price", "warning", n_null_price,
                    "Base Price was blank.",
                    "Treated the same as price-0: resolved later from the item's "
                    "own realised prices or the master list price.")
        n_zero_price = int((df["unit_price"] == 0).sum())
        n_zero_items = int(df.loc[df["unit_price"] == 0, "item_id"].nunique())
        rep.add("Price = 0 order lines", "warning", n_zero_price,
                f"{n_zero_price:,} order lines across {n_zero_items:,} products have "
                "Base Price 0. Per the business this means the price is requested "
                "from the Japanese branch per order and varies - it is NOT free.",
                "Never counted as zero revenue. Quantity is forecast normally; the "
                "unit price is recovered from the item's own realised prices, then "
                "the master list price, and only if both fail is revenue flagged "
                "PRICE_TBD.")
        n_neg_price = int((df["unit_price"] < 0).sum())
        if n_neg_price:
            rep.add("Negative unit price", "warning", n_neg_price,
                    "Base Price below zero (credit notes).",
                    "Kept - they offset revenue as intended.")

    # ------------------------------------------------------- fiscal calendar
    df = add_fiscal_columns(df, cfg["fiscal"]["start_month"])

    # -------------------------------------------------------- demand/revenue
    # Which rows count as demand, and which count as billable revenue.
    demand_statuses = cl.get("demand_statuses") or []
    revenue_statuses = cl.get("revenue_statuses") or []

    if "order_status" in df.columns:
        if demand_statuses:
            df["is_demand"] = df["order_status"].isin(demand_statuses)
        else:
            df["is_demand"] = True
        if revenue_statuses:
            df["is_revenue"] = df["order_status"].isin(revenue_statuses)
        else:
            df["is_revenue"] = True

        counted = int(df["is_revenue"].sum())
        rep.add("Order status treatment", "info", len(df) - counted,
                "The supplied data dictionary defines 'Closed' two different ways "
                "('completed and closed' vs 'will not be fulfilled or billed'). "
                f"{len(df) - counted:,} order lines are excluded from revenue.",
                "DEMAND counts every status (a cancelled order still expressed real "
                "customer demand, which is what inventory planning needs). REVENUE "
                f"counts only {', '.join(revenue_statuses)}, so we never book money "
                "that was never invoiced.")
    else:
        df["is_demand"] = True
        df["is_revenue"] = True

    df["line_revenue"] = df["quantity"] * df["unit_price"].where(df["unit_price"] > 0, np.nan)

    rep.rows_out = len(df)
    rep.stats.update({
        "date_min": df["order_date"].min(),
        "date_max": df["order_date"].max(),
        "n_products": int(df["item_id"].nunique()),
        "n_customers": int(df["customer_id"].nunique()) if "customer_id" in df else 0,
        "n_orders": int(df["so_internal_id"].nunique()) if "so_internal_id" in df else 0,
        "total_quantity": float(df["quantity"].sum()),
    })
    return df, rep


def clean_masters(
    items: pd.DataFrame, customers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Light tidy-up of the two master lists."""
    if not items.empty:
        items = items.drop_duplicates(subset="item_id", keep="first").copy()
        for c in ("production_status", "category_1", "category_2"):
            if c in items.columns:
                items[c] = items[c].fillna("Unknown").replace("", "Unknown")
        if "list_price" in items.columns:
            items["list_price"] = pd.to_numeric(items["list_price"], errors="coerce")

    if not customers.empty:
        customers = customers.drop_duplicates(subset="customer_id", keep="first").copy()
        for c in ("customer_type", "industry"):
            if c in customers.columns:
                customers[c] = customers[c].fillna("Unknown").replace("", "Unknown")

    return items, customers
