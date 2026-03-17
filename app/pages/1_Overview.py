"""
app/pages/1_Overview.py — KPIs, monthly revenue trend, revenue-by-country map
"""

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import get_engine, load_kpis, load_monthly_revenue, load_revenue_by_country

st.set_page_config(page_title="Overview", layout="wide")
st.title("Overview")

engine = get_engine()

# ── KPI cards ─────────────────────────────────────────────────────────────────
with st.spinner("Loading KPIs…"):
    kpis = load_kpis(engine)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue",     f"£{kpis['total_revenue']:,.0f}")
c2.metric("Total Customers",   f"{kpis['total_customers']:,}")
c3.metric("Total Orders",      f"{kpis['total_orders']:,}")
c4.metric("Avg Order Value",   f"£{kpis['avg_order_value']:,.2f}")

st.divider()

# ── Monthly revenue trend ──────────────────────────────────────────────────────
with st.spinner("Loading revenue trend…"):
    monthly = load_monthly_revenue(engine)

# Drop the last month — dataset ends mid-December so it's a partial month
monthly = monthly.iloc[:-1]

fig_trend = px.line(
    monthly,
    x="month",
    y="revenue",
    title="Monthly Revenue",
    labels={"month": "Month", "revenue": "Revenue (£)"},
    markers=True,
    color_discrete_sequence=["#2563EB"],
)
fig_trend.update_layout(
    hovermode="x unified",
    yaxis_tickprefix="£",
    yaxis_tickformat=",.0f",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", showline=False),
    margin=dict(t=50, b=40),
)
fig_trend.update_traces(line_width=2.5)
st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ── Revenue by country ─────────────────────────────────────────────────────────
with st.spinner("Loading country data…"):
    country_df = load_revenue_by_country(engine)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Revenue by Country")
    fig_map = px.choropleth(
        country_df,
        locations="country",
        locationmode="country names",
        color="total_revenue",
        hover_name="country",
        hover_data={
            "total_revenue":   ":,.0f",
            "customer_count":  ":,",
            "order_count":     ":,",
            "avg_order_value": ":,.2f",
        },
        color_continuous_scale="Blues",
        labels={
            "total_revenue":   "Revenue (£)",
            "customer_count":  "Customers",
            "order_count":     "Orders",
            "avg_order_value": "Avg Order (£)",
        },
        title="Total Revenue by Country",
    )
    fig_map.update_layout(
        margin=dict(t=50, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar_tickprefix="£",
        coloraxis_colorbar_tickformat=",.0f",
        geo=dict(showframe=False, showcoastlines=True, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_right:
    st.subheader("Top 10 Countries")
    top10 = country_df.head(10)[["country", "total_revenue", "customer_count"]].copy()
    top10["total_revenue"] = top10["total_revenue"].apply(lambda x: f"£{x:,.0f}")
    top10.columns = ["Country", "Revenue", "Customers"]
    st.dataframe(top10, hide_index=True, use_container_width=True)
