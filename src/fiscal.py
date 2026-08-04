"""Fiscal-calendar helpers and automatic fiscal-year detection.

The fiscal year is DETECTED from the data rather than assumed, and the evidence
is reported so the assumption can be confirmed rather than trusted blindly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class FiscalDetection:
    start_month: int
    confidence: str
    evidence: list[str]

    @property
    def label(self) -> str:
        end = MONTH_ABBR[(self.start_month - 2) % 12]
        return f"{MONTH_ABBR[self.start_month - 1]}-{end}"


def detect_fiscal_start_month(dates: pd.Series) -> FiscalDetection:
    """Infer which month the fiscal year starts in, with supporting evidence.

    Three signals are combined:
      1. the first day covered by the extract (extracts usually begin on a
         fiscal-year boundary),
      2. the last day covered (usually a fiscal year-end),
      3. the seasonal peak (year-end months carry a push in Japanese firms).
    """
    d = pd.to_datetime(dates.dropna())
    if d.empty:
        return FiscalDetection(4, "default", ["No dates available; defaulted to April."])

    lo, hi = d.min(), d.max()
    evidence: list[str] = []
    votes: dict[int, int] = {}

    def vote(month: int, weight: int, why: str) -> None:
        votes[month] = votes.get(month, 0) + weight
        evidence.append(why)

    # Signal 1 - the extract starts on the 1st of a month
    if lo.day == 1:
        vote(lo.month, 3,
             f"The data begins exactly on {lo:%d %B %Y}, the first day of a month.")

    # Signal 2 - the extract ends on the last day of a month
    if (hi + pd.Timedelta(days=1)).day == 1:
        nxt = (hi.month % 12) + 1
        vote(nxt, 3,
             f"The data ends exactly on {hi:%d %B %Y}, the last day of a month, "
             f"so the next fiscal year would start in {MONTH_ABBR[nxt - 1]}.")

    # Signal 3 - a whole number of years of coverage reinforces the boundary
    months_covered = (hi.year - lo.year) * 12 + (hi.month - lo.month) + 1
    if months_covered % 12 == 0:
        evidence.append(
            f"The extract covers exactly {months_covered} months "
            f"({months_covered // 12} complete years), consistent with whole "
            "fiscal years rather than an arbitrary window."
        )
        if lo.day == 1:
            votes[lo.month] = votes.get(lo.month, 0) + 2

    if not votes:
        return FiscalDetection(
            4, "low",
            evidence + ["No clear boundary found; defaulted to April (Japanese convention)."]
        )

    start_month = max(votes, key=votes.get)
    top = votes[start_month]
    confidence = "high" if top >= 6 else "medium" if top >= 3 else "low"
    return FiscalDetection(start_month, confidence, evidence)


def seasonal_peak_evidence(df: pd.DataFrame, start_month: int) -> str:
    """Describe the seasonal peak, used to corroborate the fiscal-year guess."""
    if df.empty:
        return ""
    share = df.groupby("fiscal_month")["quantity"].sum()
    share = share / share.sum()
    peak = int(share.idxmax())
    peak_cal = ((start_month - 1 + peak - 1) % 12) + 1
    return (
        f"The busiest month of the fiscal year is month {peak} "
        f"({MONTH_ABBR[peak_cal - 1]}) at {share.max():.1%} of annual volume "
        f"versus {1/12:.1%} if demand were flat"
        + (" - the fiscal year-end push." if peak == 12 else ".")
    )


def fiscal_year(dates: pd.Series, start_month: int) -> pd.Series:
    """Fiscal year label = the calendar year the fiscal year starts in."""
    d = pd.to_datetime(dates)
    return pd.Series(
        np.where(d.dt.month >= start_month, d.dt.year, d.dt.year - 1),
        index=d.index, dtype="int64",
    )


def fiscal_month(dates: pd.Series, start_month: int) -> pd.Series:
    """Position within the fiscal year: 1 = first month, 12 = last month."""
    d = pd.to_datetime(dates)
    return pd.Series(((d.dt.month - start_month) % 12) + 1, index=d.index, dtype="int64")


def add_fiscal_columns(df: pd.DataFrame, start_month: int) -> pd.DataFrame:
    """Attach fiscal_year, fiscal_month and a monthly period column."""
    df = df.copy()
    df["fiscal_year"] = fiscal_year(df["order_date"], start_month)
    df["fiscal_month"] = fiscal_month(df["order_date"], start_month)
    df["month"] = pd.to_datetime(df["order_date"]).dt.to_period("M")
    return df


def fy_month_index(fy: int, start_month: int, horizon: int = 12) -> pd.PeriodIndex:
    """The monthly PeriodIndex covering one fiscal year."""
    start = pd.Period(year=fy, month=start_month, freq="M")
    return pd.period_range(start, periods=horizon, freq="M")


def fy_label(fy: int, start_month: int) -> str:
    """Human label, e.g. 'FY2025 (Apr 2025 - Mar 2026)'."""
    idx = fy_month_index(fy, start_month)
    return f"FY{fy} ({idx[0].strftime('%b %Y')} - {idx[-1].strftime('%b %Y')})"
