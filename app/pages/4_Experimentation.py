"""
app/pages/4_Experimentation.py — A/B Test Calculator
Inputs: visitors + conversions for control (A) and variant (B)
Outputs: z-test, chi-squared, Bayesian posterior, power analysis
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics.ab_testing import (
    VariantData,
    run_full_analysis,
    power_analysis,
    get_posterior_samples,
)

st.set_page_config(page_title="Experimentation", layout="wide")
st.title("A/B Test Calculator")
st.markdown(
    "Enter conversion data for two variants. "
    "Get frequentist significance tests **and** a Bayesian posterior probability in one click."
)

# ── Input panel ────────────────────────────────────────────────────────────────
st.subheader("Experiment Data")

col_a, col_b, col_cfg = st.columns([2, 2, 1])

with col_a:
    st.markdown("#### Control (A)")
    a_visitors    = st.number_input("Visitors",    min_value=1, value=1000, key="av", step=100)
    a_conversions = st.number_input("Conversions", min_value=0, value=100,  key="ac", step=10)
    a_conversions = min(a_conversions, a_visitors)
    st.caption(f"Conversion rate: **{a_conversions / a_visitors:.2%}**")

with col_b:
    st.markdown("#### Variant (B)")
    b_visitors    = st.number_input("Visitors",    min_value=1, value=1000, key="bv", step=100)
    b_conversions = st.number_input("Conversions", min_value=0, value=120,  key="bc", step=10)
    b_conversions = min(b_conversions, b_visitors)
    st.caption(f"Conversion rate: **{b_conversions / b_visitors:.2%}**")

with col_cfg:
    st.markdown("#### Config")
    alpha = st.selectbox("Significance (α)", [0.05, 0.01, 0.10], index=0)
    bayes_threshold = st.slider("Bayesian threshold", 0.80, 0.99, 0.95, step=0.01)

run = st.button("▶  Run Analysis", type="primary", use_container_width=True)

if run:
    a = VariantData("A (Control)", a_visitors, a_conversions)
    b = VariantData("B (Variant)", b_visitors, b_conversions)
    results = run_full_analysis(a, b, alpha=alpha)
    z   = results["z_test"]
    chi = results["chi2"]
    bay = results["bayesian"]

    st.divider()

    # ── Headline recommendation ────────────────────────────────────────────────
    if bay.prob_b_beats_a >= bayes_threshold:
        st.success(f"✅ {bay.recommendation}")
    elif bay.prob_b_beats_a <= (1 - bayes_threshold):
        st.error(f"❌ {bay.recommendation}")
    else:
        st.warning(f"⚠️ {bay.recommendation}")

    st.divider()

    # ── Metrics row ────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Relative Lift",  f"{z.relative_lift:+.2%}")
    m2.metric("P-value (z)",    f"{z.p_value:.4f}",
              delta="significant" if z.significant else "not significant",
              delta_color="normal" if z.significant else "off")
    m3.metric("P(B > A)",       f"{bay.prob_b_beats_a:.1%}")
    m4.metric("Expected Lift",  f"{bay.expected_lift*100:+.2f}pp")

    st.divider()

    # ── Detailed results ───────────────────────────────────────────────────────
    col_freq, col_bay = st.columns(2)

    with col_freq:
        st.subheader("Frequentist Tests")
        freq_df = pd.DataFrame({
            "Test":        ["Z-test", "Chi-squared"],
            "Statistic":   [f"z = {z.z_stat:+.4f}", f"χ² = {chi.chi2_stat:.4f}"],
            "P-value":     [f"{z.p_value:.6f}", f"{chi.p_value:.6f}"],
            "Significant": ["Yes ✓" if z.significant else "No ✗",
                            "Yes ✓" if chi.significant else "No ✗"],
        })
        st.dataframe(freq_df, hide_index=True, use_container_width=True)

        st.caption(
            f"95% CI on difference (B − A): "
            f"[{z.ci_lower:+.4f}, {z.ci_upper:+.4f}]"
        )

    with col_bay:
        st.subheader("Bayesian Posterior")
        bay_df = pd.DataFrame({
            "Variant":         ["A (Control)", "B (Variant)"],
            "Posterior Mean":  [f"{bay.a_mean:.4f}", f"{bay.b_mean:.4f}"],
            "95% Credible Int": [
                f"[{bay.a_ci[0]:.4f}, {bay.a_ci[1]:.4f}]",
                f"[{bay.b_ci[0]:.4f}, {bay.b_ci[1]:.4f}]",
            ],
        })
        st.dataframe(bay_df, hide_index=True, use_container_width=True)
        st.caption(f"Prior: Beta(1, 1) — uniform. Samples: 100,000.")

    # ── Posterior distribution plot ────────────────────────────────────────────
    st.subheader("Posterior Distributions")
    samples_a, samples_b = get_posterior_samples(a, b)

    fig_post = go.Figure()
    for samples, name, color in [
        (samples_a, "A (Control)", "#6B7280"),
        (samples_b, "B (Variant)", "#2563EB"),
    ]:
        # KDE via histogram with small bins
        fig_post.add_trace(go.Histogram(
            x=samples,
            name=name,
            histnorm="probability density",
            nbinsx=80,
            marker_color=color,
            opacity=0.55,
        ))

    fig_post.update_layout(
        barmode="overlay",
        xaxis_title="Conversion Rate",
        yaxis_title="Density",
        xaxis_tickformat=".1%",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", showline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", showline=False),
        legend_title_text="Variant",
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_post, use_container_width=True)

    st.divider()

    # ── Power analysis ─────────────────────────────────────────────────────────
    st.subheader("Power Analysis")
    st.markdown("How many visitors do you need to reliably detect a given lift?")

    pa_col1, pa_col2 = st.columns(2)
    with pa_col1:
        pa_baseline = st.number_input(
            "Baseline conversion rate",
            min_value=0.001, max_value=0.999,
            value=round(a.rate, 3), step=0.005, format="%.3f",
        )
    with pa_col2:
        pa_mde = st.number_input(
            "Minimum detectable effect (absolute pp)",
            min_value=0.001, max_value=0.5,
            value=0.02, step=0.005, format="%.3f",
        )

    pa_result = power_analysis(pa_baseline, pa_mde, alpha=alpha, power=0.80)
    total_needed = pa_result.required_sample_per_variant * 2

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Required per variant",  f"{pa_result.required_sample_per_variant:,}")
    pc2.metric("Total visitors needed", f"{total_needed:,}")
    pc3.metric("Power",                 "80%")
    st.caption(
        f"To detect a **{pa_mde*100:.1f}pp lift** over a **{pa_baseline:.1%} baseline** "
        f"with 80% power at α={alpha}."
    )
