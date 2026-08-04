"""Streamlit dashboard - written for people who are not analysts.

Run with:   streamlit run app/dashboard.py

Every screen explains what it is showing in plain language, and every forecast
is shown next to the accuracy evidence that justifies it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make the project importable no matter where streamlit is launched from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_RAW, DATA_UPLOADS, OUTPUTS  # noqa: E402
from src.pipeline import load_bundle, run_pipeline  # noqa: E402

st.set_page_config(page_title="Demand & Revenue Forecast", page_icon="📦",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# Styling - restrained, high contrast, readable on a projector.
# --------------------------------------------------------------------------
st.markdown("""
<style>
  .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  div[data-testid="stMetricValue"] {font-size: 1.7rem;}
  .plain {background:#f1f5f9; border-left:4px solid #64748b; padding:0.75rem 1rem;
          border-radius:6px; margin:0.5rem 0 1rem 0; font-size:0.92rem; color:#0f172a;}
  .warn  {background:#fef3c7; border-left:4px solid #d97706; padding:0.75rem 1rem;
          border-radius:6px; margin:0.5rem 0 1rem 0; font-size:0.92rem; color:#0f172a;}
  .good  {background:#dcfce7; border-left:4px solid #16a34a; padding:0.75rem 1rem;
          border-radius:6px; margin:0.5rem 0 1rem 0; font-size:0.92rem; color:#0f172a;}
  .stTabs [data-baseweb="tab"] {font-size: 0.95rem; padding: 0.4rem 1rem;}
</style>
""", unsafe_allow_html=True)

HIST_COLOUR = "#1e3a8a"
FCST_COLOUR = "#ea580c"
GRID = "#e2e8f0"


def plain(text: str) -> None:
    st.markdown(f'<div class="plain">{text}</div>', unsafe_allow_html=True)


def warn(text: str) -> None:
    st.markdown(f'<div class="warn">{text}</div>', unsafe_allow_html=True)


def good(text: str) -> None:
    st.markdown(f'<div class="good">{text}</div>', unsafe_allow_html=True)


def thb(v: float, digits: int = 0) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if abs(v) >= 1e9:
        return f"฿{v/1e9:,.2f}bn"
    if abs(v) >= 1e6:
        return f"฿{v/1e6:,.1f}m"
    return f"฿{v:,.{digits}f}"


def units(v: float) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.0f}"


def to_ts(s: pd.Series) -> pd.Series:
    """Period -> Timestamp for charting."""
    if len(s) and isinstance(s.iloc[0], pd.Period):
        return s.dt.to_timestamp()
    return pd.to_datetime(s.astype(str))


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_bundle(token: float):
    """Cached load of the last pipeline run.

    `token` is the results file's modification time and MUST NOT start with an
    underscore - Streamlit skips hashing underscore-prefixed arguments, which
    would pin the dashboard to a stale forecast for ever after a re-run.
    """
    return load_bundle()


def bundle_token() -> float:
    p = OUTPUTS / "dashboard_bundle.pkl"
    return p.stat().st_mtime if p.exists() else 0.0


def base_chart(fig: go.Figure, height: int = 420, ytitle: str = "") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom",
                                           y=1.02, xanchor="left", x=0),
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, title=ytitle,
                     rangemode="tozero")
    return fig


def history_forecast_chart(hist_x, hist_y, fc_x, fc_y, ytitle, name_h="Actual (history)",
                           name_f="Forecast", height=420, fc_lo=None, fc_hi=None) -> go.Figure:
    """One chart with history and forecast joined, so the handover is obvious.

    When low/high estimates are supplied they are drawn as a shaded band, so the
    forecast reads as a range rather than a false-precision single line.
    """
    fig = go.Figure()
    if fc_lo is not None and fc_hi is not None and len(fc_x):
        fig.add_trace(go.Scatter(
            x=list(fc_x) + list(fc_x)[::-1],
            y=list(fc_hi) + list(fc_lo)[::-1],
            fill="toself", fillcolor="rgba(234,88,12,0.16)",
            line=dict(width=0), hoverinfo="skip",
            name="Likely range (8 in 10)"))
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, name=name_h, mode="lines+markers",
        line=dict(color=HIST_COLOUR, width=2.5), marker=dict(size=5)))
    if len(hist_x) and len(fc_x):
        # Bridge the gap so the two lines connect visually.
        fig.add_trace(go.Scatter(
            x=[list(hist_x)[-1], list(fc_x)[0]],
            y=[list(hist_y)[-1], list(fc_y)[0]],
            mode="lines", line=dict(color=FCST_COLOUR, width=2.5, dash="dot"),
            showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=fc_x, y=fc_y, name=name_f, mode="lines+markers",
        line=dict(color=FCST_COLOUR, width=2.5, dash="dash"), marker=dict(size=6)))
    if len(fc_x):
        fig.add_vrect(x0=list(fc_x)[0], x1=list(fc_x)[-1], fillcolor=FCST_COLOUR,
                      opacity=0.06, line_width=0)
    return base_chart(fig, height, ytitle)


# ==========================================================================
# Sidebar
# ==========================================================================
st.sidebar.title("📦 Forecast")
bundle = get_bundle(bundle_token())

if bundle is None:
    st.title("Demand & Revenue Forecast")
    warn("No forecast has been produced yet. Go to <b>Upload new data</b> below, "
         "or put the sales, item and customer files into <code>data/raw/</code> "
         "and press the button.")
    if st.button("▶  Run the forecast now", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        try:
            run_pipeline(data_folder=DATA_RAW,
                         on_status=lambda m, f: bar.progress(min(f, 1.0), text=m))
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Could not run the forecast: {exc}")
    st.stop()

meta = bundle["meta"]
FY = meta["forecast_fiscal_year"]
fc_all: pd.DataFrame = bundle["forecast"]
fbp: pd.DataFrame = bundle["forecast_by_product"]
hist_item: pd.DataFrame = bundle["history_item_month"]
hist_cust: pd.DataFrame = bundle["history_customer_month"]
hist_tot: pd.DataFrame = bundle["history_total"]
items: pd.DataFrame = bundle["items"]
customers: pd.DataFrame = bundle["customers"]
profile: pd.DataFrame = bundle["profile"]

st.sidebar.caption(f"**{meta['forecast_label']}**")
st.sidebar.caption(f"History: {meta['history_first_month']} → {meta['history_last_month']}")
st.sidebar.caption(f"Last run: {meta['run_at'].replace('T', ' ')}")
st.sidebar.divider()

PAGES = ["Overview", "By product", "By customer", "How accurate is this?",
         "Data quality & assumptions", "Upload new data", "Download results"]

# The chosen page is kept in the URL so a link to a specific screen can be
# shared with the team (e.g. .../?page=How+accurate+is+this%3F).
_qp = st.query_params.get("page")
_default = PAGES.index(_qp) if _qp in PAGES else 0
page = st.sidebar.radio("Go to", PAGES, index=_default, label_visibility="collapsed")
if page != _qp:
    st.query_params["page"] = page

st.sidebar.divider()
st.sidebar.caption(
    "Quantities are **expected units of demand**. Revenue counts only order "
    "statuses that historically get invoiced."
)


# ==========================================================================
# 1. OVERVIEW
# ==========================================================================
if page == "Overview":
    st.title(f"FY{FY} demand & revenue forecast")
    st.caption(meta["forecast_label"] + f" · fiscal year runs {meta['fiscal_year_label']}")

    last_fy = max(meta["history_fiscal_years"])
    ly = hist_tot[hist_tot["fiscal_year"] == last_fy]
    ly_qty, ly_rev = ly["quantity"].sum(), ly["revenue"].sum()
    f_qty = fbp["forecast_quantity"].sum()
    f_rev = fbp["forecast_revenue"].sum(skipna=True)

    q_lo, q_hi = meta.get("forecast_quantity_low"), meta.get("forecast_quantity_high")
    r_lo, r_hi = meta.get("forecast_revenue_low"), meta.get("forecast_revenue_high")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Forecast demand FY{FY}", units(f_qty),
              f"{(f_qty/ly_qty-1)*100:+.1f}% vs FY{last_fy}" if ly_qty else None)
    if q_lo:
        c1.caption(f"Likely range **{units(q_lo)} – {units(q_hi)}**")
    c2.metric(f"Forecast revenue FY{FY}", thb(f_rev),
              f"{(f_rev/ly_rev-1)*100:+.1f}% vs FY{last_fy}" if ly_rev else None)
    if r_lo:
        c2.caption(f"Likely range **{thb(r_lo)} – {thb(r_hi)}**")
    c3.metric("Products with a forecast", units(meta["n_products_forecast_nonzero"]),
              f"of {meta['n_products']:,} in the catalogue", delta_color="off")
    c4.metric("Price to be confirmed", units(meta["n_price_tbd"]),
              "products - quantity only", delta_color="off")

    plain(
        f"<b>What this page shows.</b> The company's whole order book, actual history "
        f"followed by the forecast for FY{FY}. The orange shaded area is the forecast; "
        "everything to the left of it actually happened."
        + (f"<br><br><b>Why there is a range.</b> The single number is the best "
           f"estimate, but no forecast is exact. The range is not a guess - it is how "
           f"far out this method actually was on the two past years we hid from it "
           f"and then checked. Plan against the middle number; use the range to "
           f"decide how much safety stock the risk justifies." if q_lo else "")
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Monthly demand", "Monthly revenue", "Seasonal pattern", "Sales trend"])

    with tab1:
        agg_cols = {"forecast_quantity": "sum"}
        for c in ("forecast_quantity_low", "forecast_quantity_high"):
            if c in fc_all.columns:
                agg_cols[c] = "sum"
        fc_m = fc_all.groupby("month", observed=True).agg(agg_cols).reset_index()
        # Scale the per-product band to the portfolio-level range: adding up
        # every product's worst case would assume they all happen at once.
        lo = hi = None
        if meta.get("forecast_quantity_low"):
            plo = meta["forecast_quantity_low"] / max(f_qty, 1e-9)
            phi = meta["forecast_quantity_high"] / max(f_qty, 1e-9)
            lo = fc_m["forecast_quantity"] * plo
            hi = fc_m["forecast_quantity"] * phi
        fig = history_forecast_chart(
            to_ts(hist_tot["month"]), hist_tot["quantity"],
            to_ts(fc_m["month"]), fc_m["forecast_quantity"], "Units", height=440,
            fc_lo=lo, fc_hi=hi)
        st.plotly_chart(fig, use_container_width=True)
        plain(
            "The steep drop after early 2022 and the partial recovery through FY2024 "
            "are real events in the order book, not data errors. The forecast "
            "continues from the recent level rather than from the older, higher one."
        )

    with tab2:
        fc_r = fc_all.groupby("month", observed=True)["forecast_revenue"].sum().reset_index()
        rlo = rhi = None
        if meta.get("forecast_revenue_low"):
            plo = meta["forecast_revenue_low"] / max(f_rev, 1e-9)
            phi = meta["forecast_revenue_high"] / max(f_rev, 1e-9)
            rlo = fc_r["forecast_revenue"] * plo
            rhi = fc_r["forecast_revenue"] * phi
        fig = history_forecast_chart(
            to_ts(hist_tot["month"]), hist_tot["revenue"],
            to_ts(fc_r["month"]), fc_r["forecast_revenue"], "THB", height=440,
            fc_lo=rlo, fc_hi=rhi)
        st.plotly_chart(fig, use_container_width=True)
        plain(
            "History shows revenue that was actually invoiced. The forecast applies "
            "each product's recently achieved selling price to its forecast quantity, "
            "then scales by the share of orders that historically get billed."
        )

    with tab3:
        h = hist_tot.copy()
        h["fm"] = ((pd.PeriodIndex(h["month"]).month - meta["fiscal_start_month"]) % 12) + 1
        pv = h.pivot_table(index="fm", columns="fiscal_year", values="quantity", aggfunc="sum")
        labels = [pd.Period(f"2000-{((meta['fiscal_start_month']-1+i) % 12)+1:02d}").strftime("%b")
                  for i in range(12)]
        fig = go.Figure()
        for col in pv.columns:
            fig.add_trace(go.Scatter(x=labels, y=pv[col].reindex(range(1, 13)),
                                     name=f"FY{col}", mode="lines+markers"))
        st.plotly_chart(base_chart(fig, 420, "Units"), use_container_width=True)
        plain(
            "Each line is one fiscal year. The consistent lift in the final month "
            "(March) is the fiscal year-end push, and it is one of the clues used to "
            "confirm that the fiscal year runs "
            f"{meta['fiscal_year_label']}."
        )

    with tab4:
        st.markdown("##### Year-on-year sales trend")
        fy_hist = (hist_tot.groupby("fiscal_year")
                   .agg(qty=("quantity", "sum"), rev=("revenue", "sum")).reset_index())
        fc_row = pd.DataFrame([{"fiscal_year": FY,
                                "qty": fbp["forecast_quantity"].sum(),
                                "rev": fbp["forecast_revenue"].sum(skipna=True)}])
        trend = pd.concat([fy_hist, fc_row], ignore_index=True)
        trend["qty_growth"] = trend["qty"].pct_change() * 100
        trend["rev_growth"] = trend["rev"].pct_change() * 100
        trend["What"] = ["Actual"] * len(fy_hist) + ["FORECAST"]
        show = pd.DataFrame({
            "Fiscal year": "FY" + trend["fiscal_year"].astype(str),
            "": trend["What"],
            "Demand (units)": trend["qty"].round(0),
            "Growth": trend["qty_growth"].round(1),
            "Revenue (THB)": trend["rev"].round(0),
            "Revenue growth": trend["rev_growth"].round(1),
        })
        st.dataframe(show, use_container_width=True, hide_index=True,
                     column_config={
                         "Demand (units)": st.column_config.NumberColumn(format="%.0f"),
                         "Growth": st.column_config.NumberColumn(format="%+.1f%%"),
                         "Revenue (THB)": st.column_config.NumberColumn(format="%.0f"),
                         "Revenue growth": st.column_config.NumberColumn(format="%+.1f%%"),
                     })
        plain(
            "<b>The story in one line:</b> sales were stable through FY2021-FY2022, "
            "fell sharply in FY2023, recovered strongly in FY2024, and are forecast "
            "to hold roughly flat in FY" + str(FY) + ". The FY2023 drop and FY2024 "
            "rebound are real events in the order book. Because nothing in this data "
            "explains <i>why</i> they happened, the forecast continues from the "
            "recent level rather than betting on another swing in either direction."
        )

        st.markdown("##### Which product categories are growing or shrinking")
        if "category_1" in fbp.columns and not items.empty:
            h = hist_item.merge(items[["item_id", "category_1"]], on="item_id", how="left")
            h["fy"] = h["month"].map(
                {m: (m.year if m.month >= meta["fiscal_start_month"] else m.year - 1)
                 for m in h["month"].unique()})
            last_fy = max(meta["history_fiscal_years"])
            piv = h[h["fy"].isin([last_fy - 1, last_fy])].pivot_table(
                index="category_1", columns="fy", values="quantity", aggfunc="sum").fillna(0)
            fcc = fbp.groupby("category_1", observed=True)["forecast_quantity"].sum()
            piv["Forecast FY" + str(FY)] = fcc.reindex(piv.index).fillna(0)
            piv = piv[piv.sum(axis=1) > 0].copy()
            piv["Trend vs last year"] = (
                piv["Forecast FY" + str(FY)] / piv[last_fy].replace(0, np.nan) - 1) * 100
            piv = piv.sort_values("Forecast FY" + str(FY), ascending=False)
            piv.columns = [f"FY{c}" if isinstance(c, (int, np.integer)) else c
                           for c in piv.columns]
            st.dataframe(piv.round(0), use_container_width=True,
                         column_config={"Trend vs last year":
                                        st.column_config.NumberColumn(format="%+.1f%%")})
            st.caption("Sorted by forecast size. Use this to see where growth is "
                       "expected and where stock should be reduced.")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Where the demand sits")
        if "category_1" in fbp.columns:
            g = (fbp.groupby("category_1", observed=True)
                 .agg(q=("forecast_quantity", "sum"), r=("forecast_revenue", "sum"))
                 .sort_values("q", ascending=False).head(12).reset_index())
            fig = go.Figure(go.Bar(x=g["q"], y=g["category_1"], orientation="h",
                                   marker_color=HIST_COLOUR,
                                   hovertemplate="%{y}: %{x:,.0f} units<extra></extra>"))
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(base_chart(fig, 400, "Forecast units"), use_container_width=True)

    with right:
        st.subheader("How reliable is each product?")
        cnt = profile["tier"].value_counts().sort_index()
        vol = (fbp.groupby("tier", observed=True)["forecast_quantity"].sum()
               .reindex(cnt.index).fillna(0))
        tbl = pd.DataFrame({
            "Products": cnt,
            "Forecast units": vol.round(0),
            "Share of forecast": (vol / vol.sum() * 100).round(1) if vol.sum() else 0,
        })
        st.dataframe(tbl, use_container_width=True)
        plain(
            "<b>A</b> sells almost every month - full statistical forecasting. "
            "<b>B</b> sells in some months - simple trend plus the category's seasonal "
            "shape. <b>C</b> sells rarely and unpredictably - forecast as an average "
            "rate; treat with caution. <b>D</b> had no sales at all last year and is "
            "forecast as zero."
        )

    st.divider()
    st.subheader(f"Top 15 products by forecast revenue, FY{FY}")
    top = fbp.nlargest(15, "forecast_revenue")
    show = top[[c for c in ["item_code", "category_1", "forecast_quantity",
                            "unit_price", "forecast_revenue", "model_plain",
                            "confidence", "revenue_flag"] if c in top.columns]].copy()
    show.columns = ["Product", "Category", "Forecast units", "Unit price (THB)",
                    "Forecast revenue (THB)", "Method used", "Confidence", "Price flag"]
    st.dataframe(show, use_container_width=True, hide_index=True,
                 column_config={
                     "Forecast units": st.column_config.NumberColumn(format="%.0f"),
                     "Unit price (THB)": st.column_config.NumberColumn(format="%.0f"),
                     "Forecast revenue (THB)": st.column_config.NumberColumn(format="%.0f"),
                 })


# ==========================================================================
# 2. BY PRODUCT
# ==========================================================================
elif page == "By product":
    st.title("Forecast by product")

    lookup = fbp.copy()
    lookup["label"] = lookup.apply(
        lambda r: f"{r.get('item_code', r['item_id'])}  ·  "
                  f"{r.get('category_1', '')}  ·  {r['forecast_quantity']:,.0f} units",
        axis=1)

    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        cats = ["All categories"] + sorted(
            lookup["category_1"].dropna().unique().tolist()) if "category_1" in lookup else ["All categories"]
        cat = st.selectbox("Product category", cats)
    with f2:
        tiers = ["All"] + sorted(lookup["tier"].dropna().unique().tolist())
        tier = st.selectbox("Reliability segment", tiers)

    sub = lookup
    if cat != "All categories" and "category_1" in sub:
        sub = sub[sub["category_1"] == cat]
    if tier != "All":
        sub = sub[sub["tier"] == tier]
    sub = sub.sort_values("forecast_revenue", ascending=False)

    if sub.empty:
        st.warning("No products match those filters.")
        st.stop()

    with f3:
        choice = st.selectbox(f"Product ({len(sub):,} available)", sub["label"].tolist())
    row = sub[sub["label"] == choice].iloc[0]
    iid = row["item_id"]

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Forecast demand FY{FY}", units(row["forecast_quantity"]))
    if "forecast_quantity_low" in row and pd.notna(row.get("forecast_quantity_low")):
        c1.caption(f"Likely **{units(row['forecast_quantity_low'])} – "
                   f"{units(row['forecast_quantity_high'])}**")
    c2.metric(f"Forecast revenue FY{FY}",
              thb(row["forecast_revenue"]) if pd.notna(row["forecast_revenue"]) else "Price TBD")
    if "forecast_revenue_low" in row and pd.notna(row.get("forecast_revenue_low")):
        c2.caption(f"Likely **{thb(row['forecast_revenue_low'])} – "
                   f"{thb(row['forecast_revenue_high'])}**")
    c3.metric("Unit price used",
              thb(row["unit_price"]) if pd.notna(row["unit_price"]) else "TBD")
    c4.metric("Confidence", str(row.get("confidence", "-")))

    if row.get("revenue_flag") == "PRICE_TBD":
        warn("<b>Price to be confirmed.</b> This product has never been sold at a "
             "recorded price, so no revenue figure is shown. The quantity forecast "
             "is still valid and can be used for stock planning. The price is quoted "
             "by the Japan branch for each order.")
    elif row.get("revenue_flag") == "PRICE_VARIABLE_ESTIMATED":
        warn("<b>Variable price.</b> Some past orders for this product carried no "
             "price (quoted per order by the Japan branch). Revenue here uses the "
             "average price actually achieved on the orders that <i>were</i> priced, "
             "so treat it as an estimate.")

    h = hist_item[hist_item["item_id"] == iid].sort_values("month")
    f = fc_all[fc_all["item_id"] == iid].sort_values("month")

    tab1, tab2 = st.tabs(["Demand (units)", "Revenue (THB)"])
    with tab1:
        fig = history_forecast_chart(
            to_ts(h["month"]), h["quantity"],
            to_ts(f["month"]), f["forecast_quantity"], "Units",
            fc_lo=f.get("forecast_quantity_low"), fc_hi=f.get("forecast_quantity_high"))
        st.plotly_chart(fig, use_container_width=True)
        plain(
            "The shaded band is the <b>likely range</b>. In back-testing, 8 out of 10 "
            "products in this reliability segment landed inside a band this wide. "
            "It is measured, not assumed.")
    with tab2:
        if f["forecast_revenue"].notna().any():
            fig = history_forecast_chart(to_ts(h["month"]), h["revenue"],
                                         to_ts(f["month"]), f["forecast_revenue"], "THB")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No revenue chart: this product's price is quoted per order.")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Why this number")
        method = row.get("model_plain", "-")
        st.markdown(f"**Method chosen:** {method}")
        sel = bundle["selection"]
        srow = sel[sel["item_id"] == iid]
        if not srow.empty:
            st.markdown(f"**Why:** {srow.iloc[0].get('selection_reason', '-')}")
            ru = srow.iloc[0].get("runner_up", "")
            if ru:
                st.caption(f"Next-best method tried: {ru}")
        plain(
            "Several forecasting methods were tested on past years for this product. "
            "The one shown here made the smallest error, so it was used to produce "
            f"the FY{FY} figure."
        )

    with right:
        st.subheader("How accurate was it on past years?")
        w = row.get("backtest_WAPE")
        mape_v = row.get("backtest_MAPE")
        mase_v = row.get("backtest_MASE")
        m1, m2, m3 = st.columns(3)
        m1.metric("Typical error (WAPE)", f"{w*100:,.0f}%" if pd.notna(w) else "n/a")
        m2.metric("MAPE", f"{mape_v:,.0f}%" if pd.notna(mape_v) else "n/a")
        m3.metric("vs naive (MASE)", f"{mase_v:,.2f}" if pd.notna(mase_v) else "n/a")
        plain(
            "<b>WAPE</b> is the average error as a share of the units actually sold - "
            "lower is better. <b>MASE</b> below 1.00 means this method beat a simple "
            "'same as last year' guess. MAPE is shown because it is widely quoted, "
            "but it is unreliable for products that sell in only some months."
        )

    st.subheader("Month-by-month forecast")
    cols = ["month_label", "forecast_quantity"]
    names = ["Month", "Forecast units"]
    if "forecast_quantity_low" in f.columns:
        cols += ["forecast_quantity_low", "forecast_quantity_high"]
        names += ["Low estimate", "High estimate"]
    cols += ["unit_price", "forecast_quantity_billable", "forecast_revenue"]
    names += ["Unit price (THB)", "Expected billable units", "Forecast revenue (THB)"]
    show = f[cols].copy()
    show.columns = names
    st.dataframe(show, use_container_width=True, hide_index=True,
                 column_config={
                     "Forecast units": st.column_config.NumberColumn(format="%.2f"),
                     "Low estimate": st.column_config.NumberColumn(format="%.2f"),
                     "High estimate": st.column_config.NumberColumn(format="%.2f"),
                     "Expected billable units": st.column_config.NumberColumn(format="%.2f"),
                     "Unit price (THB)": st.column_config.NumberColumn(format="%.0f"),
                     "Forecast revenue (THB)": st.column_config.NumberColumn(format="%.0f"),
                 })


# ==========================================================================
# 3. BY CUSTOMER
# ==========================================================================
elif page == "By customer":
    st.title("Forecast by customer")
    warn(
        "<b>Read this first.</b> Customer numbers are an <i>allocation</i>, not a "
        "separate forecast. Individual customers order too irregularly to forecast "
        "directly. Each product's forecast is split across its customers using their "
        "share of that product over the last 12 months. Use these figures for "
        "account planning and conversations, not as a commitment."
    )

    cf = bundle["customer_forecast"]
    if cf.empty:
        st.info("No customer-level data is available in this dataset.")
        st.stop()

    cust_tot = cf.groupby("customer_id", observed=True).agg(
        q=("forecast_quantity", "sum"), r=("forecast_revenue", "sum")).reset_index()
    if not customers.empty:
        cust_tot = cust_tot.merge(customers, on="customer_id", how="left")
    cust_tot = cust_tot.sort_values("r", ascending=False)
    cust_tot["label"] = cust_tot.apply(
        lambda x: f"{x.get('customer_name', x['customer_id'])}  ·  {thb(x['r'])}", axis=1)

    c1, c2 = st.columns([1, 2])
    with c1:
        inds = ["All industries"]
        if "industry" in cust_tot:
            inds += sorted(cust_tot["industry"].dropna().unique().tolist())
        ind = st.selectbox("Industry", inds)
    sub = cust_tot if ind == "All industries" else cust_tot[cust_tot["industry"] == ind]
    if sub.empty:
        st.warning("No customers match that filter.")
        st.stop()
    with c2:
        choice = st.selectbox(f"Customer ({len(sub):,} available)", sub["label"].tolist())

    row = sub[sub["label"] == choice].iloc[0]
    cid = row["customer_id"]

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Allocated demand FY{FY}", units(row["q"]))
    c2.metric(f"Allocated revenue FY{FY}", thb(row["r"]))
    c3.metric("Customer type", str(row.get("customer_type", "-")))
    c4.metric("Industry", str(row.get("industry", "-")))

    h = hist_cust[hist_cust["customer_id"] == cid].sort_values("month")
    f = (cf[cf["customer_id"] == cid].groupby("month", observed=True)
         .agg(q=("forecast_quantity", "sum"), r=("forecast_revenue", "sum")).reset_index())

    tab1, tab2 = st.tabs(["Demand (units)", "Revenue (THB)"])
    with tab1:
        st.plotly_chart(history_forecast_chart(
            to_ts(h["month"]), h["quantity"], to_ts(f["month"]), f["q"], "Units",
            name_f="Allocated forecast"), use_container_width=True)
    with tab2:
        st.plotly_chart(history_forecast_chart(
            to_ts(h["month"]), h["revenue"], to_ts(f["month"]), f["r"], "THB",
            name_f="Allocated forecast"), use_container_width=True)

    st.subheader("What this customer is expected to buy")
    per = (cf[cf["customer_id"] == cid].groupby("item_id", observed=True)
           .agg(q=("forecast_quantity", "sum"), r=("forecast_revenue", "sum")).reset_index())
    if not items.empty:
        per = per.merge(items[["item_id", "item_code", "category_1"]], on="item_id", how="left")
    per = per.sort_values("r", ascending=False).head(30)
    per = per[[c for c in ["item_code", "category_1", "q", "r"] if c in per.columns]]
    per.columns = ["Product", "Category", "Forecast units", "Forecast revenue (THB)"][:len(per.columns)]
    st.dataframe(per, use_container_width=True, hide_index=True,
                 column_config={
                     "Forecast units": st.column_config.NumberColumn(format="%.1f"),
                     "Forecast revenue (THB)": st.column_config.NumberColumn(format="%.0f"),
                 })

    st.subheader("Biggest customers by allocated revenue")
    topc = cust_tot.head(20)[
        [c for c in ["customer_name", "customer_type", "industry", "q", "r"]
         if c in cust_tot.columns]].copy()
    topc.columns = ["Customer", "Type", "Industry", "Forecast units",
                    "Forecast revenue (THB)"][:len(topc.columns)]
    st.dataframe(topc, use_container_width=True, hide_index=True,
                 column_config={
                     "Forecast units": st.column_config.NumberColumn(format="%.0f"),
                     "Forecast revenue (THB)": st.column_config.NumberColumn(format="%.0f"),
                 })


# ==========================================================================
# 4. ACCURACY
# ==========================================================================
elif page == "How accurate is this?":
    st.title("How accurate is this forecast?")
    plain(
        f"We do not have the real FY{FY} numbers - that is the whole point of a "
        "forecast. So accuracy is measured by <b>hiding</b> a year we do know, "
        "forecasting it with only the earlier data, and comparing against what "
        "actually happened. Two separate years were tested."
    )

    ad = bundle.get("accuracy_delivered", pd.DataFrame())
    if not ad.empty:
        st.subheader("Headline result")
        for _, r in ad.iterrows():
            st.markdown(f"**{r['Fold']} — forecasting {r['Test year']}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Units actually sold", units(r["Actual units"]))
            c2.metric("Units we would have forecast",
                      units(r["Forecast units (as delivered)"]))
            c3.metric("Total error", f"{r['Total error % (as delivered)']:+.1f}%")
            c4.metric("Typical monthly error (WAPE)",
                      f"{r['WAPE (as delivered)']*100:,.0f}%")
        st.dataframe(ad, use_container_width=True, hide_index=True)
        warn(
            "<b>Be honest about these numbers.</b> FY2023 was the year demand "
            "collapsed by 43%, and FY2024 was the year it rebounded by 25%. Neither "
            "turning point was predictable from the data that preceded it, so those "
            "total errors are close to the best anyone could do on those two years. "
            "They are the realistic worst case, not the expected error in a normal year."
        )

    st.divider()
    st.subheader("How wide is the uncertainty?")
    im = bundle.get("interval_multipliers", pd.DataFrame())
    iv = meta.get("intervals", {})
    if not im.empty:
        q_lo, q_hi = meta.get("forecast_quantity_low"), meta.get("forecast_quantity_high")
        c1, c2, c3 = st.columns(3)
        c1.metric("Low estimate", units(q_lo))
        c2.metric("Best estimate", units(meta["total_forecast_quantity"]))
        c3.metric("High estimate", units(q_hi))
        plain(
            "The range is <b>measured, not assumed</b>. For each product we took the "
            "method actually chosen for it, saw what it would have forecast for the "
            "two hidden years, and compared that against what really sold. The table "
            "below is the spread of those outcomes. A multiplier of 1.00 would mean "
            "the forecast landed exactly right."
        )
        show = im.copy()
        show.columns = ["Segment", "Low multiplier", "Typical multiplier",
                        "High multiplier", "Products measured", "Observations", "Based on"]
        st.dataframe(show, use_container_width=True, hide_index=True,
                     column_config={
                         "Low multiplier": st.column_config.NumberColumn(format="%.2f"),
                         "Typical multiplier": st.column_config.NumberColumn(format="%.2f"),
                         "High multiplier": st.column_config.NumberColumn(format="%.2f"),
                     })
        ratios = iv.get("portfolio_ratio_by_fold", {})
        if ratios:
            txt = ", ".join(f"{k}: {v:.2f}x" for k, v in ratios.items())
            warn(
                "<b>Do not add up the per-product ranges.</b> Summing every product's "
                "worst case assumes all 5,706 have a bad year simultaneously, which "
                f"would give a meaningless {units(meta.get('forecast_quantity_low_if_summed'))} "
                f"– {units(meta.get('forecast_quantity_high_if_summed'))} band. The "
                "company-wide range above instead comes from how the <i>total</i> "
                f"actually landed in each hidden year ({txt}) — which is the risk that "
                "really matters, because products move together, not independently."
            )
        st.caption(
            "Read the range as: in back-testing, 8 out of 10 products in that segment "
            "landed between the low and high multipliers. Regular sellers land close "
            "to forecast; lumpy products can easily sell nothing at all, which is why "
            "their low multiplier is 0."
        )

    st.divider()
    st.subheader("Accuracy by reliability segment")
    abt = bundle.get("accuracy_by_tier", pd.DataFrame())
    if not abt.empty:
        st.dataframe(abt, use_container_width=True, hide_index=True)
        plain(
            "Segment A carries roughly 80% of all volume and is where accuracy "
            "matters most. Segments C and D are individually unpredictable by "
            "nature - they are included so nothing is missing from the plan, "
            "not because their individual numbers are precise."
        )

    st.divider()
    st.subheader("Which methods were chosen")
    sel = bundle["selection"]
    counts = sel["model_plain"].value_counts().reset_index()
    counts.columns = ["Method", "Products"]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(counts, use_container_width=True, hide_index=True, height=380)
    with c2:
        fig = go.Figure(go.Bar(x=counts["Products"].head(12),
                               y=counts["Method"].head(12), orientation="h",
                               marker_color=HIST_COLOUR))
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(base_chart(fig, 380, "Products"), use_container_width=True)
    plain(
        "No single method was assumed to be correct. Every product was tested "
        "against every method its history could support, and the winner was kept. "
        "Simple methods win often, which is normal and healthy - a complex model "
        "was only allowed to win if it beat the simple one by a clear margin."
    )

    st.divider()
    st.subheader("Was the forecast scaled to category totals?")
    rec = meta.get("reconciliation", {})
    (good if rec.get("enabled") else plain)(rec.get("reason", "Not tested."))

    rc = meta.get("revenue_method_check") or {}
    if rc:
        st.subheader("Is the pricing method sound?")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"FY{rc['test_fiscal_year']} revenue actually invoiced",
                  thb(rc["actual_invoiced_revenue"]))
        c2.metric("Same year, rebuilt with our price logic",
                  thb(rc["revenue_from_actual_qty_and_our_prices"]))
        c3.metric("Difference", f"{rc['error_pct']:+.1f}%")
        plain(
            "This test takes the quantities that were <i>actually</i> sold last year "
            "and runs them through exactly the same pricing rules used for the "
            "forecast. Any gap is pure pricing error, separate from forecasting "
            "error. A small gap means the revenue figures rest on sound prices."
        )

    st.divider()
    st.subheader("Accuracy for every product")
    det = bundle["backtest_detail"]
    if not det.empty:
        picked = det.merge(sel[["item_id", "model"]], on=["item_id", "model"], how="inner")
        if not items.empty:
            picked = picked.merge(items[["item_id", "item_code", "category_1"]],
                                  on="item_id", how="left")
        cols = [c for c in ["item_code", "category_1", "fold", "test_fy", "model",
                            "tier", "MAE", "WAPE", "MAPE", "MASE", "actual_total",
                            "forecast_total"] if c in picked.columns]
        st.dataframe(picked[cols].head(3000), use_container_width=True, hide_index=True)
        st.caption("First 3,000 rows shown. The full table is on the Download page.")


# ==========================================================================
# 5. DATA QUALITY
# ==========================================================================
elif page == "Data quality & assumptions":
    st.title("Data quality & assumptions")
    plain(
        "Everything on this page is a decision someone made about the data. "
        "They are written down here so the team can defend the numbers, and "
        "change any of them in <code>config.yaml</code> without touching code."
    )

    good(
        "<b>Every number here comes only from your own files.</b> "
        "Sale Order History, Item and Customer — nothing else. No internet data, "
        "no market indices, no third-party or benchmark sources, and no FY"
        f"{FY} actuals. Where a method is described as learning from "
        "\"all products\", that means all products in <i>this</i> dataset."
    )

    st.subheader("The fiscal year")
    conf = meta["fiscal_detection_confidence"]
    (good if conf == "high" else warn)(
        f"<b>Detected as {meta['fiscal_year_label']}</b> (confidence: {conf}). "
        f"The forecast year FY{FY} therefore means "
        f"{meta['forecast_label'].split('(')[1].rstrip(')')}."
    )
    for e in meta.get("fiscal_detection_evidence", []):
        st.markdown(f"- {e}")
    if not meta.get("fiscal_detection_matches_config", True):
        warn("The fiscal year found in the data differs from the one set in "
             "config.yaml. The data was trusted. Please confirm this is right.")

    st.divider()
    st.subheader("Data quality checks")
    q = bundle["quality"]
    def colour(r):
        c = {"error": "background-color:#fee2e2", "warning": "background-color:#fef3c7",
             "info": "background-color:#f0fdf4"}.get(r["Severity"], "")
        return [c] * len(r)
    st.dataframe(q.style.apply(colour, axis=1), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Products whose price is quoted per order")
    plain(
        "A price of 0 in the source data does <b>not</b> mean the product is free. "
        "It means the price is requested from the Japan branch for that order and "
        "varies. These are never counted as zero revenue."
    )
    pr = bundle["prices"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Products ever priced at 0", units(int(pr["has_zero_price_history"].sum())))
    c2.metric("Price recovered from their own sales",
              units(int((pr["has_zero_price_history"] &
                         pr["price_source"].isin(["realised", "realised_all"])).sum())))
    c3.metric("Genuinely unknown (quantity only)", units(int(pr["price_is_tbd"].sum())))
    src = pr["price_source_plain"].value_counts().reset_index()
    src.columns = ["How the price was determined", "Products"]
    st.dataframe(src, use_container_width=True, hide_index=True)

    tbd = pr[pr["price_is_tbd"]]
    if not tbd.empty:
        st.markdown("**Products with no usable price at all** — quantity is forecast, "
                    "revenue is left blank rather than guessed:")
        t = tbd.merge(items[["item_id", "item_code", "category_1"]], on="item_id",
                      how="left") if not items.empty else tbd
        st.dataframe(t[[c for c in ["item_code", "category_1", "n_priced_lines_total"]
                        if c in t.columns]], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("⚠️ Open question for Mitsubishi — the VIP discount")
    v = meta.get("vip_discount_question", {})
    if v:
        c1, c2, c3 = st.columns(3)
        c1.metric("VIP customers", f"{v.get('vip_customers', 0)}")
        c2.metric("Their share of all demand",
                  f"{v.get('vip_share_of_quantity', 0)*100:.1f}%")
        sens = meta.get("revenue_if_vip_discount_37pct")
        if sens:
            gap = meta["total_forecast_revenue"] - sens
            c3.metric("Revenue at risk", thb(gap),
                      f"{gap/meta['total_forecast_revenue']*100:.1f}% of forecast",
                      delta_color="off")
        warn(
            "The data dictionary says <b>\"VIP: 37% discount from List Price\"</b>. "
            "But in the transaction data, VIP and End User lines are priced almost "
            "identically against list price (0.891 vs 0.910), and only 0.6% of VIP "
            "lines sit near 0.63x list. So the Base Price column looks like a "
            "<i>reference</i> price that does not yet have the VIP discount applied."
            "<br><br>"
            "This cannot be resolved from the data - there is no invoice-amount "
            "column to check against - so <b>no discount has been applied</b>. "
            f"If VIP invoices really are 37% lower, forecast revenue would fall from "
            f"{thb(meta['total_forecast_revenue'])} to {thb(sens) if sens else 'n/a'}."
            "<br><br>"
            "<b>Question to ask Mitsubishi:</b> <i>Is Base Price the amount actually "
            "invoiced, or a reference price before the VIP discount?</i> "
            "The demand/quantity forecast is unaffected either way."
        )

    st.divider()
    st.subheader("Safeguards applied to the forecast")
    g = fc_all.groupby("item_id")[["shortage_floor_applied", "surplus_cap_applied"]].first()
    c1, c2 = st.columns(2)
    c1.metric("Stock-shortage floor applied", units(int(g["shortage_floor_applied"].sum())))
    c2.metric("Stock-surplus cap applied", units(int(g["surplus_cap_applied"].sum())))
    plain(
        "<b>Shortage floor:</b> a product that demonstrably sold in the last 12 months "
        "is never forecast at zero for a whole year. For a lumpy product whose test "
        "year happened to contain no sales, a zero forecast scores perfectly in "
        "testing - but planning zero for a product that does sell guarantees a "
        "stockout.<br><br>"
        "<b>Surplus cap:</b> no product is forecast above 1.5x the largest annual "
        "volume it reached in the last three years. Simple methods can take one "
        "unusually large final month and project it across twelve, which quietly "
        "turns into surplus stock."
    )

    st.divider()
    st.subheader("Key modelling assumptions")
    st.markdown(f"""
1. **Fiscal year is {meta['fiscal_year_label']}**, detected from the data and
   corroborated by the March peak in demand.
2. **Demand counts every order status; revenue counts only invoiced statuses.**
   The supplied data dictionary defines "Closed" two contradictory ways. A closed
   order still expressed real customer demand, which is what stock planning needs,
   but it should not be booked as revenue.
3. **Exact duplicate order lines were kept.** Evidence indicates they are genuine
   repeated ERP lines: only part of each order repeats, they are spread evenly
   across all four years, and 65% carry quantity 1.
4. **Discontinued products are not automatically excluded.** 171 of them still
   sold last year, so they are forecast and flagged instead of dropped.
5. **Monthly quantities are kept fractional.** Rounding each month to a whole
   number deleted 1,922 real slow-moving products from the plan.
6. **Large orders are never deleted as outliers** - they are real demand.
7. **Products with no sales at all last year are forecast as zero** and labelled,
   rather than being silently omitted.
    """)
    st.caption(f"Total pipeline runtime: {meta['runtime_seconds']}s · "
               f"{meta['rows_out']:,} order lines used of {meta['rows_in']:,} supplied.")


# ==========================================================================
# 6. UPLOAD
# ==========================================================================
elif page == "Upload new data":
    st.title("Upload new data and re-run the forecast")
    plain(
        "Upload newer files here and the whole process runs again from scratch: "
        "cleaning, testing every method on the new history, choosing the best one "
        "per product, and producing a fresh forecast. <b>No code changes needed.</b>"
    )

    st.subheader("What the files need to look like")
    with st.expander("Required columns (click to open)", expanded=False):
        st.markdown("""
**Sales history** *(required)* — one row per order line
`Sales Order Date`, `Item Internal ID`, `Quantity` are essential.
`SO Internal ID`, `Sales Order Number`, `Sales Order Status`,
`Customer Internal ID`, `Company Name`, `Item Code`, `Base Price` are used if present.

**Item master** *(optional but recommended)*
`Item Internal ID`, `Item Code`, `Production Status`, `Product Category 1`,
`Product Category 2`, `List price`.

**Customer master** *(optional)*
`Customer Internal ID`, `Company Name`, `Customer Type`, `FA Sub Industries`.

Both **.csv** and **.xlsx** are accepted. Column names are matched loosely, so
extra spaces, different capitalisation and small renames are handled
automatically — you will see exactly how each column was matched before anything runs.
        """)

    c1, c2, c3 = st.columns(3)
    with c1:
        up_sales = st.file_uploader("Sales history (required)", type=["csv", "xlsx", "xls"])
    with c2:
        up_item = st.file_uploader("Item master (optional)", type=["csv", "xlsx", "xls"])
    with c3:
        up_cust = st.file_uploader("Customer master (optional)", type=["csv", "xlsx", "xls"])

    ups = {k: v for k, v in
           {"sales": up_sales, "item": up_item, "customer": up_cust}.items() if v}

    if ups:
        st.divider()
        st.subheader("Check before running")
        from src.ingest import load_table
        ok = True
        for table, fh in ups.items():
            df, rep = load_table(fh, table)
            with st.expander(f"**{table}** — {rep.rows:,} rows from `{rep.source}`",
                             expanded=True):
                if rep.errors:
                    ok = False
                    for e in rep.errors:
                        st.error(e)
                if rep.missing_required:
                    ok = False
                    st.error("Missing essential columns: " + ", ".join(rep.missing_required))
                mapped = pd.DataFrame(
                    [{"Needed": k, "Found in your file": v} for k, v in rep.mapped.items()])
                if not mapped.empty:
                    st.dataframe(mapped, use_container_width=True, hide_index=True)
                if rep.missing_optional:
                    st.warning("Not found (will be skipped): " +
                               ", ".join(rep.missing_optional))
                if rep.unmapped_source_cols:
                    st.caption("Columns in your file that were not used: " +
                               ", ".join(map(str, rep.unmapped_source_cols)))
                if not df.empty:
                    st.dataframe(df.head(5), use_container_width=True)

        if "sales" not in ups:
            st.info("Only the sales history can be replaced on its own — the item "
                    "and customer files already on disk will be reused.")

        st.divider()
        keep = st.checkbox(
            "Also save these files into data/raw so they become the new default", value=True)
        if st.button("▶  Run the forecast on this data", type="primary", disabled=not ok):
            if keep:
                DATA_UPLOADS.mkdir(parents=True, exist_ok=True)
                for table, fh in ups.items():
                    fh.seek(0)
                    (DATA_UPLOADS / fh.name).write_bytes(fh.read())
                    fh.seek(0)
            bar = st.progress(0.0, text="Starting...")
            try:
                for fh in ups.values():
                    fh.seek(0)
                run_pipeline(data_folder=DATA_RAW, uploads=ups,
                             on_status=lambda m, f: bar.progress(min(f, 1.0), text=m))
                st.cache_data.clear()
                good("Done. Every page now shows the new forecast.")
                st.balloons()
                st.rerun()
            except Exception as exc:
                st.error(f"The run failed: {exc}")
    else:
        st.info("Choose at least a sales history file to begin.")

    st.divider()
    st.subheader("Or just re-run on the files already in data/raw")
    st.caption(f"Folder: `{DATA_RAW}`")
    if st.button("↻  Re-run on existing files"):
        bar = st.progress(0.0, text="Starting...")
        try:
            run_pipeline(data_folder=DATA_RAW,
                         on_status=lambda m, f: bar.progress(min(f, 1.0), text=m))
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"The run failed: {exc}")


# ==========================================================================
# 7. DOWNLOADS
# ==========================================================================
elif page == "Download results":
    st.title("Download results")
    plain("Every file below is produced by the last run and is safe to share.")

    files = [
        (f"FY{FY}_forecast_by_product_month.csv",
         "**The main deliverable** — every product, every month, quantity, unit "
         "price, revenue and the price flag."),
        (f"FY{FY}_forecast_summary_by_product.csv",
         "One row per product: the full-year forecast, method used and accuracy."),
        (f"FY{FY}_forecast_by_customer.csv",
         "Forecast allocated to customers (allocation, not a separate forecast)."),
        ("backtest_accuracy_as_delivered.csv",
         "Headline accuracy of the complete pipeline, safeguards included."),
        ("backtest_accuracy_summary.csv",
         "Accuracy of the selected models on each test year."),
        ("backtest_accuracy_by_segment.csv", "Accuracy split by reliability segment."),
        ("backtest_accuracy_by_product.csv", "Accuracy of the chosen method per product."),
        ("backtest_all_models_scored.csv",
         "Every method scored on every product — the full evidence base."),
        ("model_selection_by_product.csv", "Which method was chosen for each product and why."),
        ("data_quality_report.csv", "Every data issue found and what was done about it."),
        ("product_profile.csv", "Demand behaviour of every product."),
        ("unit_price_resolution.csv", "How each product's unit price was determined."),
        ("prediction_interval_multipliers.csv",
         "The measured low/high multipliers behind every forecast range."),
        ("run_metadata.json", "Full record of the run, including every setting used."),
    ]

    for name, desc in files:
        p = OUTPUTS / name
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**{name}**")
            st.caption(desc)
        with c2:
            if p.exists():
                st.download_button("Download", p.read_bytes(), file_name=name,
                                   mime="text/csv", key=name, use_container_width=True)
            else:
                st.caption("not produced")
        st.divider()
