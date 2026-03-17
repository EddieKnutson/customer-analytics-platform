"""
app/db.py — Shared DB connection and data-loading functions for Streamlit pages.

All query functions return pandas DataFrames.
Backend-agnostic: works with PostgreSQL (local dev) and SQLite (Streamlit Cloud).
Cohort retention is computed in pandas since DATE_TRUNC / AGE don't exist in SQLite.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import db_url


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_engine():
    return create_engine(db_url(), echo=False)


def is_sqlite(engine) -> bool:
    return engine.dialect.name == "sqlite"


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW PAGE QUERIES
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_kpis(_engine) -> dict:
    if is_sqlite(_engine):
        q = """
            SELECT
                ROUND(SUM(ii.line_total), 0)                         AS total_revenue,
                COUNT(DISTINCT i.customer_id)                        AS total_customers,
                COUNT(DISTINCT i.invoice_no)                         AS total_orders,
                ROUND(SUM(ii.line_total) / COUNT(DISTINCT i.invoice_no), 2) AS avg_order_value
            FROM invoices i
            JOIN invoice_items ii ON ii.invoice_no = i.invoice_no
            WHERE i.is_cancelled = 0 AND ii.quantity > 0 AND ii.unit_price > 0
        """
    else:
        q = """
            SELECT
                ROUND(SUM(ii.line_total)::NUMERIC, 0)                 AS total_revenue,
                COUNT(DISTINCT i.customer_id)                         AS total_customers,
                COUNT(DISTINCT i.invoice_no)                          AS total_orders,
                ROUND(SUM(ii.line_total)::NUMERIC
                    / COUNT(DISTINCT i.invoice_no), 2)                AS avg_order_value
            FROM invoices i
            JOIN invoice_items ii ON ii.invoice_no = i.invoice_no
            WHERE i.is_cancelled = FALSE AND ii.quantity > 0 AND ii.unit_price > 0
        """
    with _engine.connect() as conn:
        row = conn.execute(text(q)).fetchone()
    return {
        "total_revenue":    float(row[0]),
        "total_customers":  int(row[1]),
        "total_orders":     int(row[2]),
        "avg_order_value":  float(row[3]),
    }


@st.cache_data(ttl=3600)
def load_monthly_revenue(_engine) -> pd.DataFrame:
    """Monthly revenue trend — returned as DataFrame with cols: month, revenue."""
    if is_sqlite(_engine):
        q = """
            SELECT
                strftime('%Y-%m-01', i.invoice_date) AS month,
                ROUND(SUM(ii.line_total), 0)          AS revenue
            FROM invoices i
            JOIN invoice_items ii ON ii.invoice_no = i.invoice_no
            WHERE i.is_cancelled = 0 AND ii.quantity > 0 AND ii.unit_price > 0
            GROUP BY 1
            ORDER BY 1
        """
    else:
        q = """
            SELECT
                DATE_TRUNC('month', i.invoice_date)       AS month,
                ROUND(SUM(ii.line_total)::NUMERIC, 0)     AS revenue
            FROM invoices i
            JOIN invoice_items ii ON ii.invoice_no = i.invoice_no
            WHERE i.is_cancelled = FALSE AND ii.quantity > 0 AND ii.unit_price > 0
            GROUP BY 1
            ORDER BY 1
        """
    df = pd.read_sql(q, _engine)
    df["month"] = pd.to_datetime(df["month"])
    df["revenue"] = df["revenue"].astype(float)
    return df


@st.cache_data(ttl=3600)
def load_revenue_by_country(_engine) -> pd.DataFrame:
    """Revenue, customer count, and order count per country."""
    if is_sqlite(_engine):
        # On SQLite (deployed), this is a materialized table from export_sqlite.py
        try:
            return pd.read_sql("SELECT * FROM revenue_by_country ORDER BY total_revenue DESC", _engine)
        except Exception:
            pass
        # fallback: inline aggregation
        q = """
            SELECT c.country,
                   ROUND(SUM(ii.line_total), 2) AS total_revenue,
                   COUNT(DISTINCT i.customer_id)  AS customer_count,
                   COUNT(DISTINCT i.invoice_no)   AS order_count,
                   ROUND(SUM(ii.line_total) / COUNT(DISTINCT i.invoice_no), 2) AS avg_order_value
            FROM invoices i
            JOIN invoice_items ii ON ii.invoice_no = i.invoice_no
            JOIN customers c ON c.customer_id = i.customer_id
            WHERE i.is_cancelled = 0 AND ii.quantity > 0 AND ii.unit_price > 0
            GROUP BY c.country
            ORDER BY total_revenue DESC
        """
    else:
        q = "SELECT * FROM vw_revenue_by_country"
    return pd.read_sql(q, _engine)


# ─────────────────────────────────────────────────────────────────────────────
# RFM SEGMENTATION PAGE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_segments(_engine) -> pd.DataFrame:
    """Load customer_segments table — works on both backends (real table)."""
    return pd.read_sql("SELECT * FROM customer_segments", _engine)


# ─────────────────────────────────────────────────────────────────────────────
# COHORT RETENTION PAGE
# Computed entirely in pandas — avoids DATE_TRUNC / AGE SQL functions
# that don't exist in SQLite.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_cohort_retention(_engine) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        cohort_month, months_since_join, customers, cohort_size, retention_rate

    Computed in pandas so it works on both PostgreSQL and SQLite.
    """
    # Load only the columns we need
    invoices = pd.read_sql(
        "SELECT customer_id, invoice_date FROM invoices WHERE is_cancelled = %s"
        % ("FALSE" if not is_sqlite(_engine) else "0"),
        _engine,
    )
    invoices["invoice_date"] = pd.to_datetime(invoices["invoice_date"])

    # Cohort month = first purchase month per customer
    invoices["activity_month"] = invoices["invoice_date"].dt.to_period("M")
    cohort_map = (
        invoices.groupby("customer_id")["activity_month"]
        .min()
        .rename("cohort_month")
    )
    invoices = invoices.join(cohort_map, on="customer_id")

    # Months since join
    invoices["months_since_join"] = (
        invoices["activity_month"] - invoices["cohort_month"]
    ).apply(lambda x: x.n)

    # Cohort sizes
    cohort_sizes = (
        invoices.groupby("cohort_month")["customer_id"]
        .nunique()
        .rename("cohort_size")
    )

    # Retention matrix
    retention = (
        invoices.groupby(["cohort_month", "months_since_join"])["customer_id"]
        .nunique()
        .rename("customers")
        .reset_index()
    )
    retention = retention.join(cohort_sizes, on="cohort_month")
    retention["retention_rate"] = (
        retention["customers"] / retention["cohort_size"]
    ).round(4)

    # Convert Period to timestamp for Plotly
    retention["cohort_month"] = retention["cohort_month"].dt.to_timestamp()
    return retention.sort_values(["cohort_month", "months_since_join"])
