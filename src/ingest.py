"""Data ingestion: read the three source tables from CSV or Excel, map their
columns onto a canonical schema, and validate them.

Designed so a non-technical user can upload a new file later with slightly
different column spellings (extra spaces, different case, minor renames) and
have it still work. Every mapping decision is reported back to the user.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------
# Canonical schema. Each canonical name lists the spellings we accept for it.
# Matching is case-insensitive and ignores spaces/underscores, so
# "Item Internal ID", " item_internal_id " and "ItemInternalID" all match.
# --------------------------------------------------------------------------
SALES_SCHEMA: dict[str, list[str]] = {
    "so_internal_id":   ["SO Internal ID", "Sales Order Internal ID", "SO ID"],
    "order_date":       ["Sales Order Date", "Sales Sheet Date", "Order Date", "Date"],
    "order_number":     ["Sales Order Number", "Sales Sheet Number", "Order Number"],
    "order_status":     ["Sales Order Status", "Sales Sheet Status", "Status"],
    "customer_id":      ["Customer Internal ID", "Internal ID Customer", "Customer ID"],
    "customer_name":    ["Company Name", "Customer Name"],
    "item_id":          ["Item Internal ID", "Internal ID Item", "Item ID"],
    "item_code":        ["Item Code", "Item Name", "Product Code"],
    "unit_price":       ["Base Price", "Price", "Unit Price"],
    "quantity":         ["Quantity", "Qty"],
}

ITEM_SCHEMA: dict[str, list[str]] = {
    "item_id":          ["Item Internal ID", "Internal ID Item", "Item ID"],
    "item_code":        ["Item Code", "Item Name", "Product Code"],
    "production_status": ["Production Status", "Status"],
    "category_1":       ["Product Category 1", "Category 1", "Product Family"],
    "category_2":       ["Product Category 2", "Category 2"],
    "list_price":       ["List price", "List Price", "Price", "Standard Price"],
}

CUSTOMER_SCHEMA: dict[str, list[str]] = {
    "customer_id":      ["Customer Internal ID", "Internal ID Customer", "Customer ID"],
    "customer_name":    ["Company Name", "Customer Name"],
    "customer_type":    ["Customer Type", "Type"],
    "industry":         ["FA Sub Industries", "FA Sub Industry", "Industry"],
}

# Columns without which the pipeline genuinely cannot run.
REQUIRED = {
    "sales": ["order_date", "item_id", "quantity"],
    "item": ["item_id"],
    "customer": ["customer_id"],
}

# Default filenames looked for in data/raw. Matching is fuzzy (see find_source_files).
DEFAULT_FILE_HINTS = {
    "sales": ["sale order history", "sales order history", "sales", "order"],
    "item": ["item", "product"],
    "customer": ["customer", "client"],
}


@dataclass
class LoadReport:
    """Everything that happened during ingestion, for display to the user."""
    table: str
    source: str
    rows: int
    mapped: dict[str, str] = field(default_factory=dict)     # canonical -> original
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    unmapped_source_cols: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.missing_required


def _norm(name: str) -> str:
    """Normalise a column name for tolerant matching."""
    return "".join(str(name).lower().split()).replace("_", "").replace("-", "")


def _read_any(source: Any, sheet_hint: str | None = None) -> pd.DataFrame:
    """Read a CSV or Excel file from a path or an uploaded file-like object."""
    name = getattr(source, "name", str(source))
    suffix = Path(name).suffix.lower()

    if suffix in (".csv", ".txt"):
        # Try utf-8 first, then fall back for Excel-exported Thai/Japanese CSVs.
        for enc in ("utf-8-sig", "utf-8", "cp874", "cp932", "latin-1"):
            try:
                if hasattr(source, "seek"):
                    source.seek(0)
                return pd.read_csv(source, encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue
        raise ValueError(f"Could not decode CSV file: {name}")

    if suffix in (".xlsx", ".xls", ".xlsm"):
        if hasattr(source, "seek"):
            source.seek(0)
            data = io.BytesIO(source.read())
        else:
            data = source
        xls = pd.ExcelFile(data)
        sheet = xls.sheet_names[0]
        if sheet_hint:
            for s in xls.sheet_names:
                if _norm(sheet_hint) in _norm(s):
                    sheet = s
                    break
        return pd.read_excel(xls, sheet_name=sheet)

    raise ValueError(
        f"Unsupported file type '{suffix}' for {name}. Please upload .csv or .xlsx."
    )


def _map_columns(
    df: pd.DataFrame, schema: dict[str, list[str]], table: str, source_name: str
) -> tuple[pd.DataFrame, LoadReport]:
    """Rename source columns to canonical names using the alias table."""
    rep = LoadReport(table=table, source=source_name, rows=len(df))

    lookup = {_norm(c): c for c in df.columns}
    used_source: set[str] = set()
    rename: dict[str, str] = {}

    for canonical, aliases in schema.items():
        for alias in aliases:
            src = lookup.get(_norm(alias))
            if src is not None and src not in used_source:
                rename[src] = canonical
                used_source.add(src)
                rep.mapped[canonical] = src
                break

    out = df.rename(columns=rename)

    for canonical in schema:
        if canonical not in rep.mapped:
            if canonical in REQUIRED.get(table, []):
                rep.missing_required.append(canonical)
            else:
                rep.missing_optional.append(canonical)

    rep.unmapped_source_cols = [c for c in df.columns if c not in used_source]
    # Keep only canonical columns that we actually found.
    keep = [c for c in schema if c in rep.mapped]
    return out[keep], rep


def _coerce_types(df: pd.DataFrame, table: str, rep: LoadReport) -> pd.DataFrame:
    """Force each canonical column to its expected dtype, recording failures."""
    df = df.copy()

    if "order_date" in df.columns:
        parsed = pd.to_datetime(df["order_date"], errors="coerce", dayfirst=True)
        # If day-first parsing produced many NaT, retry without the hint.
        if parsed.isna().mean() > 0.5:
            parsed = pd.to_datetime(df["order_date"], errors="coerce")
        bad = int(parsed.isna().sum())
        if bad:
            rep.errors.append(f"{bad:,} rows have an unreadable order date and were dropped.")
        df["order_date"] = parsed

    for col in ("item_id", "customer_id", "so_internal_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ("quantity", "unit_price", "list_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("item_code", "customer_name", "order_status", "order_number",
                "production_status", "category_1", "category_2",
                "customer_type", "industry"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    return df


def load_table(
    source: Any, table: str, sheet_hint: str | None = None
) -> tuple[pd.DataFrame, LoadReport]:
    """Load and validate one table. `table` is 'sales', 'item' or 'customer'."""
    schema = {"sales": SALES_SCHEMA, "item": ITEM_SCHEMA, "customer": CUSTOMER_SCHEMA}[table]
    source_name = getattr(source, "name", str(source))
    try:
        raw = _read_any(source, sheet_hint=sheet_hint or table)
    except Exception as exc:
        rep = LoadReport(table=table, source=source_name, rows=0)
        rep.errors.append(f"Could not read the file: {exc}")
        return pd.DataFrame(), rep

    df, rep = _map_columns(raw, schema, table, source_name)
    df = _coerce_types(df, table, rep)
    rep.rows = len(df)
    return df, rep


def find_source_files(folder: Path) -> dict[str, Path | None]:
    """Locate the three source files in a folder by fuzzy filename match."""
    folder = Path(folder)
    found: dict[str, Path | None] = {"sales": None, "item": None, "customer": None}
    if not folder.exists():
        return found

    candidates = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls", ".xlsm")
        and not p.name.startswith("~$")
    ]
    # Match most specific hints first so "Sale Order History" doesn't get
    # claimed by the "item"/"customer" matchers.
    for table in ("sales", "item", "customer"):
        for hint in DEFAULT_FILE_HINTS[table]:
            for p in candidates:
                if hint in p.stem.lower() and p not in found.values():
                    found[table] = p
                    break
            if found[table]:
                break
    return found


def load_all(
    folder: Path | None = None, uploads: dict[str, Any] | None = None
) -> tuple[dict[str, pd.DataFrame], list[LoadReport]]:
    """Load all three tables, preferring uploaded files over files on disk.

    `uploads` maps 'sales'/'item'/'customer' to file-like objects (as provided
    by Streamlit's file_uploader). Anything not uploaded is read from `folder`.
    """
    uploads = uploads or {}
    tables: dict[str, pd.DataFrame] = {}
    reports: list[LoadReport] = []

    on_disk = find_source_files(folder) if folder else {}

    for table in ("sales", "item", "customer"):
        src = uploads.get(table) or on_disk.get(table)
        if src is None:
            rep = LoadReport(table=table, source="(not provided)", rows=0)
            if table == "sales":
                rep.errors.append(
                    "No sales history file found. This file is required."
                )
            else:
                rep.missing_optional.append("(whole file)")
            reports.append(rep)
            tables[table] = pd.DataFrame()
            continue

        df, rep = load_table(src, table)
        tables[table] = df
        reports.append(rep)

    return tables, reports
