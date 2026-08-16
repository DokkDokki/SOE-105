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
  .stTabs [data-baseweb="tab"] {font-size: 0.95rem; padding: 0.4rem 1rem;}
</style>
""", unsafe_allow_html=True)

HIST_COLOUR = "#1e3a8a"
FCST_COLOUR = "#ea580c"
GRID = "#e2e8f0"
# Charts render on a forced white canvas, so text and axes are pinned to dark
# values rather than inherited from the viewer's theme.
INK = "#0f172a"
AXIS = "#94a3b8"


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
    """Apply the shared chart styling.

    The plot background is forced to white, so every text colour must be forced
    dark to match. Without this, a viewer using the dark theme gets Plotly's
    inherited light-grey font painted onto a white canvas, which renders the
    legend, axis labels and tick values effectively invisible.
    """
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=INK, size=13), bgcolor="rgba(255,255,255,0.85)"),
        # No chart-level title: the surrounding tab/section already names the
        # chart, and a title dict without text renders as the literal string
        # "undefined" in Plotly.
        font=dict(size=13, color=INK),
        hoverlabel=dict(font=dict(color=INK), bgcolor="white",
                        bordercolor=GRID),
    )
    axis_style = dict(
        showgrid=True, gridcolor=GRID, zeroline=False,
        linecolor=AXIS, tickcolor=AXIS,
        tickfont=dict(color=INK, size=12),
        title_font=dict(color=INK, size=13),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(title=ytitle, rangemode="tozero", **axis_style)
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
            name="Expected range (8 in 10 products)"))
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
    st.info("No forecast has yet been generated. Use Data Upload & Refresh, or place the source files in data/raw and select Generate forecast.")
    if st.button("Generate forecast", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        try:
            run_pipeline(data_folder=DATA_RAW,
                         on_status=lambda m, f: bar.progress(min(f, 1.0), text=m))
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"The forecast could not be generated: {exc}")
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

PAGES = ["Overview", "Product Analysis", "Customer Analysis",
         "Inventory Planning", "Accuracy & Performance Metrics",
         "Data Quality & Assumptions", "Data Upload & Refresh",
         "Downloads & Exports"]

# The chosen page is kept in the URL so a link to a specific screen can be
# shared with the team (e.g. .../?page=How+accurate+is+this%3F).
_qp = st.query_params.get("page")
_default = PAGES.index(_qp) if _qp in PAGES else 0
page = st.sidebar.radio("Go to", PAGES, index=_default, label_visibility="collapsed")
if page != _qp:
    st.query_params["page"] = page

st.sidebar.divider()


# ==========================================================================
# 1. OVERVIEW
# ==========================================================================
if page == "Overview":
    st.title(f"FY{FY} Demand and Revenue Forecast")
    st.caption(meta["forecast_label"] + f" · fiscal year runs {meta['fiscal_year_label']}")

    last_fy = max(meta["history_fiscal_years"])
    ly = hist_tot[hist_tot["fiscal_year"] == last_fy]
    ly_qty, ly_rev = ly["quantity"].sum(), ly["revenue"].sum()
    f_qty = fbp["forecast_quantity"].sum()
    f_rev = fbp["forecast_revenue"].sum(skipna=True)

    q_lo, q_hi = meta.get("forecast_quantity_low"), meta.get("forecast_quantity_high")
    r_lo, r_hi = meta.get("forecast_revenue_low"), meta.get("forecast_revenue_high")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Forecast demand, FY{FY}", units(f_qty),
              f"{(f_qty/ly_qty-1)*100:+.1f}% vs FY{last_fy}" if ly_qty else None)
    if q_lo:
        c1.caption(f"Expected range: **{units(q_lo)} – {units(q_hi)}**")
    c2.metric(f"Forecast revenue, FY{FY}", thb(f_rev),
              f"{(f_rev/ly_rev-1)*100:+.1f}% vs FY{last_fy}" if ly_rev else None)
    if r_lo:
        c2.caption(f"Expected range: **{thb(r_lo)} – {thb(r_hi)}**")
    c3.metric("Products forecast", units(meta["n_products_forecast_nonzero"]),
              f"of {meta['n_products']:,} in the catalogue", delta_color="off")
    c4.metric("Pricing to be confirmed", units(meta["n_price_tbd"]),
              "products; quantity only", delta_color="off")


    tab1, tab2, tab3, tab4 = st.tabs(
        ["Monthly Demand", "Monthly Revenue", "Seasonal Profile", "Sales Trend"])

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

    with tab4:
        st.markdown("##### Year-on-Year Performance")
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

        st.markdown("##### Category Growth and Decline")
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

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Forecast Demand by Product Category")
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
        st.subheader("Forecast Reliability by Product Segment")
        cnt = profile["tier"].value_counts().sort_index()
        vol = (fbp.groupby("tier", observed=True)["forecast_quantity"].sum()
               .reindex(cnt.index).fillna(0))
        tbl = pd.DataFrame({
            "Products": cnt,
            "Forecast units": vol.round(0),
            "Share of forecast": (vol / vol.sum() * 100).round(1) if vol.sum() else 0,
        })
        st.dataframe(tbl, use_container_width=True)

    st.divider()
    st.subheader(f"Leading Products by Forecast Revenue, FY{FY}")
    top = fbp.nlargest(15, "forecast_revenue")
    show = top[[c for c in ["item_code", "category_1", "forecast_quantity",
                            "unit_price", "forecast_revenue", "model_plain",
                            "confidence", "revenue_flag"] if c in top.columns]].copy()
    show.columns = ["Product", "Category", "Forecast units", "Unit price (THB)",
                    "Forecast revenue (THB)", "Method applied", "Confidence",
                    "Pricing flag"]
    st.dataframe(show, use_container_width=True, hide_index=True,
                 column_config={
                     "Forecast units": st.column_config.NumberColumn(format="%.0f"),
                     "Unit price (THB)": st.column_config.NumberColumn(format="%.0f"),
                     "Forecast revenue (THB)": st.column_config.NumberColumn(format="%.0f"),
                 })


# ==========================================================================
# 2. BY PRODUCT
# ==========================================================================
elif page == "Product Analysis":
    st.title("Product Analysis")

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
        st.warning("No products correspond to the selected filters.")
        st.stop()

    with f3:
        choice = st.selectbox(f"Product ({len(sub):,} available)", sub["label"].tolist())
    row = sub[sub["label"] == choice].iloc[0]
    iid = row["item_id"]

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Forecast demand, FY{FY}", units(row["forecast_quantity"]))
    if "forecast_quantity_low" in row and pd.notna(row.get("forecast_quantity_low")):
        c1.caption(f"Expected range: **{units(row['forecast_quantity_low'])} – "
                   f"{units(row['forecast_quantity_high'])}**")
    c2.metric(f"Forecast revenue, FY{FY}",
              thb(row["forecast_revenue"]) if pd.notna(row["forecast_revenue"])
              else "Pricing TBC")
    if "forecast_revenue_low" in row and pd.notna(row.get("forecast_revenue_low")):
        c2.caption(f"Expected range: **{thb(row['forecast_revenue_low'])} – "
                   f"{thb(row['forecast_revenue_high'])}**")
    c3.metric("Unit price used",
              thb(row["unit_price"]) if pd.notna(row["unit_price"]) else "TBD")
    c4.metric("Confidence", str(row.get("confidence", "-")))

    h = hist_item[hist_item["item_id"] == iid].sort_values("month")
    f = fc_all[fc_all["item_id"] == iid].sort_values("month")

    tab1, tab2 = st.tabs(["Demand (units)", "Revenue (THB)"])
    with tab1:
        fig = history_forecast_chart(
            to_ts(h["month"]), h["quantity"],
            to_ts(f["month"]), f["forecast_quantity"], "Units",
            fc_lo=f.get("forecast_quantity_low"), fc_hi=f.get("forecast_quantity_high"))
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        if f["forecast_revenue"].notna().any():
            fig = history_forecast_chart(to_ts(h["month"]), h["revenue"],
                                         to_ts(f["month"]), f["forecast_revenue"], "THB")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No revenue chart is available, as pricing for this product is quoted "
                    "on a per-order basis.")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Basis of Forecast")
        method = row.get("model_plain", "-")
        st.markdown(f"**Method selected:** {method}")
        sel = bundle["selection"]
        srow = sel[sel["item_id"] == iid]
        if not srow.empty:
            st.markdown(f"**Rationale:** {srow.iloc[0].get('selection_reason', '-')}")
            ru = srow.iloc[0].get("runner_up", "")
            if ru:
                st.caption(f"Next-best method evaluated: {ru}")

    with right:
        st.subheader("Historical Accuracy")
        w = row.get("backtest_WAPE")
        mape_v = row.get("backtest_MAPE")
        mase_v = row.get("backtest_MASE")
        m1, m2, m3 = st.columns(3)
        m1.metric("WAPE", f"{w*100:,.0f}%" if pd.notna(w) else "n/a")
        m2.metric("MAPE", f"{mape_v:,.0f}%" if pd.notna(mape_v) else "n/a")
        m3.metric("MASE", f"{mase_v:,.2f}" if pd.notna(mase_v) else "n/a")

    st.subheader("Monthly Forecast Detail")
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
elif page == "Customer Analysis":
    st.title("Customer Analysis")

    cf = bundle["customer_forecast"]
    if cf.empty:
        st.info("No customer-level data is available within this dataset.")
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
        st.warning("No customers correspond to the selected filter.")
        st.stop()
    with c2:
        choice = st.selectbox(f"Customer ({len(sub):,} available)", sub["label"].tolist())

    row = sub[sub["label"] == choice].iloc[0]
    cid = row["customer_id"]

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Allocated demand, FY{FY}", units(row["q"]))
    c2.metric(f"Allocated revenue, FY{FY}", thb(row["r"]))
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

    st.subheader("Forecast Composition by Product")
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

    st.subheader("Leading Customers by Allocated Revenue")
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
elif page == "Inventory Planning":
    st.title("Inventory Planning")
    fr = bundle.get("frontier", pd.DataFrame())


    if fr.empty:
        st.info("No frontier is available. Please re-run the forecast.")
    else:
        # ---------------- the frontier chart ----------------
        st.subheader("Service Level Against Inventory Held")
        fig = go.Figure()
        colours = {"This forecasting system": FCST_COLOUR,
                   "Naive (last year's average)": AXIS}
        for label, g in fr.groupby("Approach", sort=False):
            g = g.sort_values("Average stock held (units)")
            fig.add_trace(go.Scatter(
                x=g["Average stock held (units)"], y=g["Service level %"],
                mode="lines+markers+text",
                text=[f"{f:.2g}x" for f in g["Stock factor"]],
                textposition="top center",
                textfont=dict(size=10, color=INK),
                name=label,
                line=dict(color=colours.get(label, HIST_COLOUR), width=2.5),
                marker=dict(size=7),
                hovertemplate=("stock factor %{text}<br>average stock "
                               "%{x:,.0f} units<br>service level "
                               "%{y:.1f}%<extra>" + label + "</extra>"),
            ))
        fig = base_chart(fig, 460, "Demand served (%)")
        fig.update_xaxes(title="Average stock held (units)")
        fig.update_yaxes(rangemode="normal")
        st.plotly_chart(fig, use_container_width=True)

        # ---------------- pick a target service level ----------------
        st.subheader("Selecting an Operating Point")
        from src.inventory import factor_for_service, recommend_stock
        target = st.slider(
            "Target service level (% of demand served from stock)",
            min_value=50, max_value=85, value=70, step=5)
        rec = factor_for_service(fr, float(target))
        if rec:
            c1, c2, c3 = st.columns(3)
            c1.metric("Required stock factor", f"{rec['factor']:.2f}x")
            c2.metric("Service level achieved in simulation",
                      f"{rec['service']:.1f}%")
            c3.metric("Average stock this implied", units(rec["stock"]))
            if not rec["achievable"]:
                st.caption("The requested service level was not reached by any policy evaluated; the figures shown are the highest point on the curve.")
            recdf = recommend_stock(
                fc_all[["item_id", "item_code", "month_label",
                        "forecast_quantity"]].rename(
                    columns={"month_label": "month"}),
                rec["factor"])
            recdf = recdf.rename(columns={
                "item_id": "Item Internal ID", "item_code": "Item Code",
                "month": "Month", "forecast_quantity": "Forecast quantity",
                "recommended_stock": f"Recommended stock ({rec['factor']:.2f}x)"})
            if "event_exposure_p90_month" in fbp.columns:
                recdf = recdf.merge(
                    fbp[["item_id", "event_exposure_p90_month"]].rename(columns={
                        "item_id": "Item Internal ID",
                        "event_exposure_p90_month":
                            "Event exposure (P90 single-month demand, units)"}),
                    on="Item Internal ID", how="left")
            st.download_button(
                f"Download recommended stock levels ({rec['factor']:.2f}x, CSV)",
                recdf.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"FY{FY}_recommended_stock_{rec['factor']:.2f}x.csv",
                mime="text/csv")

        # ---------------- event exposure ----------------
        if "event_exposure_p90_month" in fbp.columns:
            st.subheader("Event Exposure")
            ex = fbp[fbp["event_exposure_p90_month"] > 0].nlargest(
                12, "event_exposure_p90_month")
            cols_ok = [c for c in ["item_code", "category_1",
                                   "forecast_quantity",
                                   "event_exposure_p90_month"] if c in ex.columns]
            show_ex = ex[cols_ok].rename(columns={
                "item_code": "Product", "category_1": "Category",
                "forecast_quantity": "Annual forecast (units)",
                "event_exposure_p90_month": "P90 single-month demand (units)"})
            st.dataframe(show_ex, use_container_width=True, hide_index=True,
                         column_config={
                             "Annual forecast (units)":
                                 st.column_config.NumberColumn(format="%.0f"),
                             "P90 single-month demand (units)":
                                 st.column_config.NumberColumn(format="%.0f"),
                         })

        # ---------------- the underlying numbers ----------------
        st.subheader("Simulation Detail")
        show = fr.copy()
        st.dataframe(show, use_container_width=True, hide_index=True,
                     column_config={
                         "Stock factor": st.column_config.NumberColumn(format="%.2f"),
                         "Service level %": st.column_config.NumberColumn(format="%.1f%%"),
                         "Average stock held (units)":
                             st.column_config.NumberColumn(format="%.0f"),
                         "Demand missed (units)":
                             st.column_config.NumberColumn(format="%.0f"),
                     })

elif page == "Accuracy & Performance Metrics":
    st.title("Accuracy and Performance Metrics")

    # ---- 1. LEAD WITH THE BENCHMARK: is this better than the obvious alternative?
    bm = bundle.get("benchmark", pd.DataFrame())
    ad = bundle.get("accuracy_delivered", pd.DataFrame())

    if not bm.empty:
        st.subheader("Performance Against Simpler Alternatives")
        for fold, g in bm.groupby("Fold", sort=True):
            ty = g["Test year"].iloc[0]
            imp = g["_improvement"].iloc[0]
            st.markdown(f"**{fold} — forecast period {ty}**")
            show = g[["Method", "WAPE", "Total error %"]].copy()
            show["WAPE"] = show["WAPE"].round(3)
            show["Total error %"] = show["Total error %"].round(1)
            st.dataframe(
                show, use_container_width=True, hide_index=True,
                column_config={
                    "WAPE": st.column_config.NumberColumn(format="%.3f"),
                    "Total error %": st.column_config.NumberColumn(format="%+.1f%%"),
                })

    # ---- 1b. THE DECISION LADDER
    lv = bundle.get("accuracy_levels", pd.DataFrame())
    if not lv.empty:
        st.divider()
        st.subheader("Accuracy by Decision Level")
        show = lv.drop(columns=["Test year"])
        st.dataframe(show, use_container_width=True, hide_index=True,
                     column_config={
                         "WAPE": st.column_config.NumberColumn(format="%.3f"),
                         "Accuracy %": st.column_config.ProgressColumn(
                             format="%.1f%%", min_value=0, max_value=100),
                     })

    # ---- 2. CONTEXT BEFORE THE RAW FIGURES
    st.divider()
    st.subheader("Trading Conditions in the Test Periods")

    # ---- 3. THE RAW FIGURES, PROPERLY FORMATTED
    if not ad.empty:
        st.subheader("Summary of Back-Test Results")
        for _, r in ad.iterrows():
            st.markdown(f"**{r['Fold']} — forecast period {r['Test year']}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Actual units sold", units(r["Actual units"]))
            c2.metric("Forecast units",
                      units(r["Forecast units (as delivered)"]))
            c3.metric("Total error", f"{r['Total error % (as delivered)']:+.1f}%")
            c4.metric("WAPE", f"{r['WAPE (as delivered)']:.3f}")
        tidy = ad.rename(columns={
            "Forecast units (models only)": "Forecast (models only)",
            "Forecast units (as delivered)": "Forecast (as delivered)",
            "Total error % (models only)": "Error % (models only)",
            "Total error % (as delivered)": "Error % (as delivered)",
            "WAPE (models only)": "WAPE (models only)",
            "WAPE (as delivered)": "WAPE (as delivered)",
        })
        st.dataframe(
            tidy, use_container_width=True, hide_index=True,
            column_config={
                "Actual units": st.column_config.NumberColumn(format="%.0f"),
                "Forecast (models only)": st.column_config.NumberColumn(format="%.0f"),
                "Forecast (as delivered)": st.column_config.NumberColumn(format="%.0f"),
                "Error % (models only)": st.column_config.NumberColumn(format="%+.1f%%"),
                "Error % (as delivered)": st.column_config.NumberColumn(format="%+.1f%%"),
                "WAPE (models only)": st.column_config.NumberColumn(format="%.3f"),
                "WAPE (as delivered)": st.column_config.NumberColumn(format="%.3f"),
            })

    st.divider()
    st.subheader("Prediction Intervals")
    im = bundle.get("interval_multipliers", pd.DataFrame())
    iv = meta.get("intervals", {})
    if not im.empty:
        q_lo, q_hi = meta.get("forecast_quantity_low"), meta.get("forecast_quantity_high")
        c1, c2, c3 = st.columns(3)
        c1.metric("Lower bound", units(q_lo))
        c2.metric("Central estimate", units(meta["total_forecast_quantity"]))
        c3.metric("Upper bound", units(q_hi))
        show = im.copy()
        show.columns = ["Segment", "Lower multiplier", "Median multiplier",
                        "Upper multiplier", "Products measured", "Observations",
                        "Basis"]
        st.dataframe(show, use_container_width=True, hide_index=True,
                     column_config={
                         "Lower multiplier": st.column_config.NumberColumn(format="%.2f"),
                         "Median multiplier": st.column_config.NumberColumn(format="%.2f"),
                         "Upper multiplier": st.column_config.NumberColumn(format="%.2f"),
                     })
        ratios = iv.get("portfolio_ratio_by_fold", {})
        if ratios:
            txt = ", ".join(f"{k}: {v:.2f}x" for k, v in ratios.items())

    st.divider()
    st.subheader("Accuracy by Product Segment")
    abt = bundle.get("accuracy_by_tier", pd.DataFrame())
    if not abt.empty:
        st.dataframe(abt, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Model Selection Distribution")
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

    st.divider()
    st.subheader("Hierarchical Reconciliation")
    rec = meta.get("reconciliation", {})

    rc = meta.get("revenue_method_check") or {}
    if rc:
        st.subheader("Validation of Pricing Methodology")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"FY{rc['test_fiscal_year']} revenue invoiced",
                  thb(rc["actual_invoiced_revenue"]))
        c2.metric("Reconstructed using this pricing logic",
                  thb(rc["revenue_from_actual_qty_and_our_prices"]))
        c3.metric("Variance", f"{rc['error_pct']:+.1f}%")

    st.divider()
    st.subheader("Per-Product Accuracy")
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


# ==========================================================================
# 5. DATA QUALITY
# ==========================================================================
elif page == "Data Quality & Assumptions":
    st.title("Data Quality and Assumptions")


    st.subheader("Fiscal Year Determination")
    conf = meta["fiscal_detection_confidence"]
    st.caption(f"Determined as {meta['fiscal_year_label']} (confidence: {conf}).")
    if not meta.get("fiscal_detection_matches_config", True):
        st.warning("The fiscal year determined from the data differs from the one "
                   "specified in config.yaml; the data was treated as authoritative.")

    st.divider()
    st.subheader("Data Quality Checks")
    q = bundle["quality"]
    st.dataframe(q, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Products Subject to Variable Pricing")
    pr = bundle["prices"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Products with any zero-priced order", units(int(pr["has_zero_price_history"].sum())))
    c2.metric("Price recovered from sales history",
              units(int((pr["has_zero_price_history"] &
                         pr["price_source"].isin(["realised", "realised_all"])).sum())))
    c3.metric("Price indeterminate (quantity only)", units(int(pr["price_is_tbd"].sum())))
    src = pr["price_source_plain"].value_counts().reset_index()
    src.columns = ["Basis of price determination", "Products"]
    st.dataframe(src, use_container_width=True, hide_index=True)

    tbd = pr[pr["price_is_tbd"]]
    if not tbd.empty:
        st.markdown("**Products for which no usable price exists.** Quantity is "
                    "forecast; revenue is left blank rather than estimated:")
        t = tbd.merge(items[["item_id", "item_code", "category_1"]], on="item_id",
                      how="left") if not items.empty else tbd
        st.dataframe(t[[c for c in ["item_code", "category_1", "n_priced_lines_total"]
                        if c in t.columns]], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Forecast Safeguards")
    g = fc_all.groupby("item_id")[["shortage_floor_applied", "surplus_cap_applied"]].first()
    c1, c2 = st.columns(2)
    c1.metric("Shortage floor applied", units(int(g["shortage_floor_applied"].sum())))
    c2.metric("Surplus cap applied", units(int(g["surplus_cap_applied"].sum())))

    st.caption(f"Total processing time: {meta['runtime_seconds']} seconds. "
               f"{meta['rows_out']:,} order lines used of {meta['rows_in']:,} supplied.")


# ==========================================================================
# 6. UPLOAD
# ==========================================================================
elif page == "Data Upload & Refresh":
    st.title("Data Upload and Refresh")
    st.subheader("File Requirements")
    with st.expander("Required columns", expanded=False):
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
        st.subheader("Validation Summary")
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
                    st.error("Required columns not found: " + ", ".join(rep.missing_required))
                mapped = pd.DataFrame(
                    [{"Required field": k, "Matched column": v} for k, v in rep.mapped.items()])
                if not mapped.empty:
                    st.dataframe(mapped, use_container_width=True, hide_index=True)
                if rep.missing_optional:
                    st.warning("Not found; these fields will be omitted: " +
                               ", ".join(rep.missing_optional))
                if rep.unmapped_source_cols:
                    st.caption("Columns present in the file but not used: " +
                               ", ".join(map(str, rep.unmapped_source_cols)))
                if not df.empty:
                    st.dataframe(df.head(5), use_container_width=True)

        if "sales" not in ups:
            st.info("The sales history may be replaced independently; the item and "
                    "customer files already present will be reused.")

        st.divider()
        keep = st.checkbox(
            "Retain these files as the new default dataset", value=True)
        if st.button("Generate forecast from this data", type="primary", disabled=not ok):
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
                st.success("Complete. All pages now reflect the updated forecast.")
                st.balloons()
                st.rerun()
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
    else:
        st.info("Please select at least a sales history file to proceed.")

    st.divider()
    st.subheader("Re-run Using Existing Files")
    st.caption(f"Folder: `{DATA_RAW}`")
    if st.button("Re-run forecast"):
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
elif page == "Downloads & Exports":
    st.title("Downloads and Exports")

    files = [
        (f"FY{FY}_forecast_by_product_month.csv",
         "**Principal deliverable.** Every product and month, with quantity, unit "
         "price, revenue and pricing flag."),
        (f"FY{FY}_forecast_summary_by_product.csv",
         "One row per product: full-year forecast, method applied and accuracy."),
        (f"FY{FY}_forecast_by_customer.csv",
         "Forecast allocated to customers. An allocation, not an independent forecast."),
        ("backtest_accuracy_as_delivered.csv",
         "Headline accuracy of the complete pipeline, inclusive of safeguards."),
        ("backtest_accuracy_summary.csv",
         "Accuracy of the selected models on each test year."),
        ("backtest_accuracy_by_segment.csv", "Accuracy split by reliability segment."),
        ("backtest_accuracy_by_product.csv", "Accuracy of the chosen method per product."),
        ("backtest_all_models_scored.csv",
         "Every method scored against every product. The complete evidence base."),
        ("model_selection_by_product.csv", "The method selected for each product, and the rationale."),
        ("data_quality_report.csv", "Each data issue identified and the action taken."),
        ("product_profile.csv", "Demand characteristics of every product."),
        ("unit_price_resolution.csv", "How each product's unit price was determined."),
        ("accuracy_by_decision_level.csv",
         "The same forecast scored at every level a decision is taken at."),
        ("inventory_service_frontier.csv",
         "The service-level / stock-held trade-off curve from the inventory "
         "simulation, including the naive comparison."),
        ("prediction_interval_multipliers.csv",
         "The measured lower and upper multipliers underlying each forecast range."),
        ("run_metadata.json", "Complete record of the run, including all configuration settings applied."),
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
                st.caption("not generated")
        st.divider()
