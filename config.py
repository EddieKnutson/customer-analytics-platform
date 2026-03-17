"""
config.py — Central configuration for Customer Analytics & Experimentation Platform
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
SQL_DIR    = BASE_DIR / "sql"
OUTPUT_DIR = BASE_DIR / "output"

# Raw UCI Online Retail dataset (download separately — see README)
RAW_DATA_PATH = DATA_DIR / "online_retail.xlsx"

# SQLite DB used for Streamlit Cloud deployment
SQLITE_DB_PATH = DATA_DIR / "retail.db"

# ── PostgreSQL (local development) ─────────────────────────────────────────────
PG_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "database": os.getenv("PG_DB",   "retail_analytics"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

# SQLAlchemy connection string — PostgreSQL
def pg_url() -> str:
    c = PG_CONFIG
    return f"postgresql+psycopg2://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['database']}"

# SQLAlchemy connection string — SQLite (Streamlit Cloud fallback)
def sqlite_url() -> str:
    return f"sqlite:///{SQLITE_DB_PATH}"

# ── ETL settings ───────────────────────────────────────────────────────────────
# Set USE_POSTGRES=True when running locally with PostgreSQL
USE_POSTGRES = os.getenv("USE_POSTGRES", "false").lower() == "true"

def db_url() -> str:
    return pg_url() if USE_POSTGRES else sqlite_url()

# ── Analysis settings ──────────────────────────────────────────────────────────
RFM_SNAPSHOT_DATE = None   # None = max(InvoiceDate) + 1 day; or set "2011-12-10"
N_CLUSTERS = 5             # K-Means default; elbow method will suggest best k
RANDOM_STATE = 42

# Segment label mapping — update after running elbow method
SEGMENT_LABELS = {
    # cluster_id (int) -> business label
    # Populated after initial clustering run
}
