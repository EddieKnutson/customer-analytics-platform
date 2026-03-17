"""
app/main.py — Customer Analytics Platform — Home / Entry Point

Run with:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Customer Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Customer Analytics & Experimentation Platform")
st.caption("UCI Online Retail dataset · ~500K transactions · UK e-commerce retailer · 2010–2011")

st.divider()

st.markdown("### Why this exists")
st.markdown(
    """
    Most e-commerce teams are sitting on years of transaction data but struggle to answer
    three basic questions:

    - **Who are our best customers** — and are we keeping them?
    - **Are new customers coming back** — or buying once and disappearing?
    - **When we run a promotion or test a new feature** — did it actually work?

    This platform answers all three, end-to-end, in one place.
    """
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("#### Overview")
    st.markdown(
        "Top-line KPIs, monthly revenue trend, and a world map showing "
        "which countries drive the most revenue. The starting point for any business review."
    )

with col2:
    st.markdown("#### RFM Segmentation")
    st.markdown(
        "Clusters customers into segments — **Champions, Loyal, Potential, Lapsed** — "
        "based on how recently they bought, how often, and how much. "
        "Tells you who to reward, who to re-engage, and who to write off."
    )

with col3:
    st.markdown("#### Cohort Retention")
    st.markdown(
        "Groups customers by the month they first purchased and tracks "
        "what percentage came back each month after. "
        "The clearest way to see whether retention is improving or deteriorating over time."
    )

with col4:
    st.markdown("#### Experimentation")
    st.markdown(
        "A/B test calculator that runs a frequentist z-test, chi-squared test, "
        "and a Bayesian model in parallel. Outputs a plain-English recommendation "
        "and tells you how many visitors you need before running the test."
    )

st.divider()

st.markdown("### Stack")
st.markdown(
    "`PostgreSQL` · `SQLite` · `SQLAlchemy` · `Pandas` · `Scikit-learn` · `SciPy` · `Streamlit` · `Plotly`"
)
st.caption("Source: UCI ML Repository — Online Retail (Daqing Chen et al., 2015)")
