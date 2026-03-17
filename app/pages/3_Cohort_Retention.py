"""
app/pages/3_Cohort_Retention.py — Monthly cohort retention heatmap
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import get_engine, load_cohort_retention

st.set_page_config(page_title="Cohort Retention", layout="wide")
st.title("Cohort Retention Analysis")

st.markdown(
    """
    Each row is a **cohort** — the month a customer made their first purchase.
    Each column is **months since joining**.
    Values show the percentage of that cohort still purchasing in that period.
    """
)

engine = get_engine()

with st.spinner("Computing cohort retention…"):
    df = load_cohort_retention(engine)

# ── Controls ───────────────────────────────────────────────────────────────────
max_month = int(df["months_since_join"].max())
col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    max_months_shown = st.slider(
        "Max months to show",
        min_value=3,
        max_value=min(max_month, 12),
        value=min(max_month, 12),
    )
with col_ctrl2:
    min_cohort_size = st.slider(
        "Min cohort size (filter small cohorts)",
        min_value=1,
        max_value=100,
        value=10,
    )

# ── Filter and pivot ───────────────────────────────────────────────────────────
df_filtered = df[
    (df["months_since_join"] <= max_months_shown)
    & (df["cohort_size"] >= min_cohort_size)
].copy()

df_filtered["cohort_label"] = df_filtered["cohort_month"].dt.strftime("%Y-%m")

pivot = df_filtered.pivot_table(
    index="cohort_label",
    columns="months_since_join",
    values="retention_rate",
    aggfunc="first",
)
pivot = pivot.sort_index()

# Cohort sizes for y-axis labels
cohort_sizes = (
    df_filtered[df_filtered["months_since_join"] == 0]
    .set_index("cohort_label")["cohort_size"]
    .to_dict()
)
y_labels = [f"{c}  (n={cohort_sizes.get(c, '?'):,})" for c in pivot.index]

# ── Heatmap ────────────────────────────────────────────────────────────────────
z = pivot.values * 100  # convert to percentages for display
x_labels = [f"Month {int(m)}" for m in pivot.columns]

text_annotations = []
for row in z:
    row_text = []
    for val in row:
        row_text.append(f"{val:.1f}%" if not pd.isna(val) else "")
    text_annotations.append(row_text)

fig = go.Figure(data=go.Heatmap(
    z=z,
    x=x_labels,
    y=y_labels,
    text=text_annotations,
    texttemplate="%{text}",
    textfont=dict(size=10),
    colorscale="Blues",
    zmin=0,
    zmax=100,
    colorbar=dict(title="Retention %", ticksuffix="%"),
    hoverongaps=False,
    hovertemplate=(
        "Cohort: %{y}<br>"
        "%{x}<br>"
        "Retention: %{z:.1f}%<extra></extra>"
    ),
))

fig.update_layout(
    title="Monthly Cohort Retention",
    xaxis_title="Months Since First Purchase",
    yaxis_title="Cohort (First Purchase Month)",
    height=max(400, len(pivot) * 32 + 100),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=60, b=60, l=180, r=60),
)

st.plotly_chart(fig, use_container_width=True)

# ── Average retention curve ────────────────────────────────────────────────────
st.subheader("Average Retention by Month")
avg_retention = df_filtered.groupby("months_since_join")["retention_rate"].mean().reset_index()
avg_retention["retention_pct"] = avg_retention["retention_rate"] * 100

import plotly.express as px
fig_curve = px.line(
    avg_retention,
    x="months_since_join",
    y="retention_pct",
    markers=True,
    labels={"months_since_join": "Months Since First Purchase", "retention_pct": "Avg Retention (%)"},
    color_discrete_sequence=["#2563EB"],
)
fig_curve.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", showline=False, ticksuffix="%", range=[0, 105]),
    margin=dict(t=30, b=40),
)
fig_curve.add_hline(y=50, line_dash="dash", line_color="#DC2626", opacity=0.5,
                    annotation_text="50% retention", annotation_position="right")
st.plotly_chart(fig_curve, use_container_width=True)
