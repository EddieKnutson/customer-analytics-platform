"""
app/pages/2_RFM_Segmentation.py — RFM scatter, radar chart, segment table
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import get_engine, load_segments

st.set_page_config(page_title="RFM Segmentation", layout="wide")
st.title("RFM Customer Segmentation")

engine = get_engine()

with st.spinner("Loading segment data…"):
    df = load_segments(engine)

# ── Segment colour palette ─────────────────────────────────────────────────────
SEGMENT_COLORS = {
    "Champions": "#16A34A",
    "Loyal":     "#2563EB",
    "Potential": "#D97706",
    "New":       "#7C3AED",
    "At-Risk":   "#DC2626",
    "Lapsed":    "#6B7280",
}
df["color"] = df["segment"].map(SEGMENT_COLORS).fillna("#94A3B8")

# ── Segment filter ─────────────────────────────────────────────────────────────
all_segments = sorted(df["segment"].unique())
selected = st.multiselect(
    "Filter segments",
    options=all_segments,
    default=all_segments,
)
filtered = df[df["segment"].isin(selected)]

st.divider()

# ── Scatter: Recency vs Monetary, sized by Frequency ─────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Recency vs Spend")
    # Cap axes at 95th percentile to prevent outliers from squashing the main cluster
    x_cap = int(filtered["recency_days"].quantile(0.95))
    y_cap = float(filtered["monetary"].quantile(0.95))
    plot_df = filtered[
        (filtered["recency_days"] <= x_cap) & (filtered["monetary"] <= y_cap)
    ]
    n_hidden = len(filtered) - len(plot_df)

    fig_scatter = px.scatter(
        plot_df,
        x="recency_days",
        y="monetary",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_data={"customer_id": True, "frequency": True},
        labels={
            "recency_days": "Days Since Last Purchase",
            "monetary":     "Total Spend (£)",
            "frequency":    "Orders",
            "segment":      "Segment",
        },
        opacity=0.65,
        title="Recency vs Spend",
    )
    fig_scatter.update_traces(marker_size=5)
    fig_scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Segment",
        margin=dict(t=50, b=40),
        yaxis_tickprefix="£",
        yaxis_tickformat=",.0f",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", showline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", showline=False),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    if n_hidden > 0:
        st.caption(f"{n_hidden} outliers hidden (above 95th percentile on recency or spend).")

# ── Radar chart: mean R/F/M per segment ───────────────────────────────────────
with col2:
    st.subheader("Avg RFM Profile per Segment")
    radar_data = (
        filtered.groupby("segment")[["r_score", "f_score", "m_score"]]
        .mean()
        .reset_index()
    )

    categories = ["Recency Score", "Frequency Score", "Monetary Score"]
    fig_radar = go.Figure()

    for _, row in radar_data.iterrows():
        values = [row["r_score"], row["f_score"], row["m_score"]]
        values_closed = values + [values[0]]  # close the polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["segment"],
            line_color=SEGMENT_COLORS.get(row["segment"], "#94A3B8"),
            opacity=0.7,
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                gridcolor="rgba(255,255,255,0.12)",
                linecolor="rgba(255,255,255,0.12)",
            ),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.12)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend_title_text="Segment",
        margin=dict(t=50, b=40),
        title="Mean R/F/M Quintile Scores by Segment",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── Segment summary table ──────────────────────────────────────────────────────
st.subheader("Segment Summary")
summary = (
    filtered.groupby("segment")
    .agg(
        Customers=("customer_id", "count"),
        Pct=("customer_id", lambda x: len(x) / len(df)),
        Avg_Recency=("recency_days", "mean"),
        Avg_Frequency=("frequency", "mean"),
        Avg_Spend=("monetary", "mean"),
        Total_Revenue=("monetary", "sum"),
    )
    .reset_index()
    .sort_values("Total_Revenue", ascending=False)
)
summary["Pct"]           = summary["Pct"].apply(lambda x: f"{x:.1%}")
summary["Avg_Recency"]   = summary["Avg_Recency"].apply(lambda x: f"{x:.0f}d")
summary["Avg_Frequency"] = summary["Avg_Frequency"].apply(lambda x: f"{x:.1f}")
summary["Avg_Spend"]     = summary["Avg_Spend"].apply(lambda x: f"£{x:,.0f}")
summary["Total_Revenue"] = summary["Total_Revenue"].apply(lambda x: f"£{x:,.0f}")
summary.columns = ["Segment", "Customers", "% of Base", "Avg Recency",
                   "Avg Orders", "Avg Spend", "Total Revenue"]
st.dataframe(summary, hide_index=True, use_container_width=True)

# ── Segment size bar chart ─────────────────────────────────────────────────────
st.subheader("Customers per Segment")
count_df = (
    filtered.groupby("segment")["customer_id"]
    .count()
    .reset_index()
    .sort_values("customer_id", ascending=False)
)
fig_bar = px.bar(
    count_df,
    x="segment",
    y="customer_id",
    color="segment",
    color_discrete_map=SEGMENT_COLORS,
    labels={"segment": "Segment", "customer_id": "Customers"},
    text="customer_id",
)
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(
    showlegend=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", showline=False),
    margin=dict(t=30, b=40),
    yaxis_title="Number of Customers",
)
st.plotly_chart(fig_bar, use_container_width=True)
