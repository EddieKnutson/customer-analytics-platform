"""
etl/export_sqlite.py — Export PostgreSQL → SQLite for Streamlit Cloud deployment.

What it does:
  1. Reads all tables + materialized view data from PostgreSQL
  2. Creates a fresh SQLite DB at data/retail.db
  3. Writes all tables — skipping generated columns (line_total)
     so SQLite can recompute them via its own GENERATED ALWAYS AS clause
  4. Materializes PostgreSQL views into plain tables so the Streamlit app
     can query them on SQLite without DATE_TRUNC / AGE / :: syntax

Usage:
    USE_POSTGRES=true PG_USER=postgres PG_PASSWORD=... python -m etl.export_sqlite

Output:
    data/retail.db   (~50–80 MB depending on SQLite page size)
"""

import sys
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import pg_url, sqlite_url, SQLITE_DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── SQLite schema for base tables ─────────────────────────────────────────────
# Mirrors schema.sql but SQLite-compatible.
# line_total is GENERATED ALWAYS AS so we don't insert it — SQLite computes it.

SQLITE_SCHEMA = """
DROP TABLE IF EXISTS invoice_items;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS customer_segments;
DROP TABLE IF EXISTS revenue_by_country;

CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    country       VARCHAR(100) NOT NULL
);

CREATE TABLE products (
    stock_code    VARCHAR(20) PRIMARY KEY,
    description   TEXT
);

CREATE TABLE invoices (
    invoice_no    VARCHAR(20) PRIMARY KEY,
    customer_id   INTEGER REFERENCES customers(customer_id),
    invoice_date  TIMESTAMP NOT NULL,
    is_cancelled  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE invoice_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_no    VARCHAR(20) REFERENCES invoices(invoice_no),
    stock_code    VARCHAR(20) REFERENCES products(stock_code),
    quantity      INTEGER NOT NULL,
    unit_price    REAL NOT NULL,
    line_total    REAL GENERATED ALWAYS AS (quantity * unit_price) STORED
);

CREATE TABLE customer_segments (
    customer_id     INTEGER PRIMARY KEY,
    country         VARCHAR(100),
    recency_days    INTEGER,
    frequency       INTEGER,
    monetary        REAL,
    r_score         INTEGER,
    f_score         INTEGER,
    m_score         INTEGER,
    rfm_score       INTEGER,
    kmeans_cluster  INTEGER,
    hc_cluster      INTEGER,
    segment         VARCHAR(50)
);

-- Materialized view tables (replaces PostgreSQL views for SQLite)
CREATE TABLE revenue_by_country (
    country          VARCHAR(100),
    total_revenue    REAL,
    customer_count   INTEGER,
    order_count      INTEGER,
    avg_order_value  REAL
);

CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices(customer_id);
CREATE INDEX IF NOT EXISTS idx_invoices_date     ON invoices(invoice_date);
CREATE INDEX IF NOT EXISTS idx_items_invoice     ON invoice_items(invoice_no);
CREATE INDEX IF NOT EXISTS idx_items_stock       ON invoice_items(stock_code);
"""

CHUNK = 10_000


def setup_sqlite(sqlite_engine) -> None:
    with sqlite_engine.begin() as conn:
        for stmt in SQLITE_SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    log.info("SQLite schema created.")


def export_table(pg_engine, sqlite_engine, table: str,
                 drop_cols: list = None, chunk: int = CHUNK) -> None:
    drop_cols = drop_cols or []
    log.info("Exporting %s …", table)

    # Count rows first
    with pg_engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    log.info("  %s rows", f"{count:,}")

    # Stream in chunks
    offset = 0
    while offset < count:
        df = pd.read_sql(
            f"SELECT * FROM {table} LIMIT {chunk} OFFSET {offset}",
            pg_engine,
        )
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df.to_sql(table, sqlite_engine, if_exists="append", index=False)
        offset += chunk
        if offset % (chunk * 10) == 0 or offset >= count:
            log.info("  … %s / %s", f"{min(offset, count):,}", f"{count:,}")


def export_view(pg_engine, sqlite_engine, view: str, target_table: str) -> None:
    """Read a PostgreSQL view and write it as a plain table in SQLite."""
    log.info("Materialising %s → %s …", view, target_table)
    df = pd.read_sql(f"SELECT * FROM {view}", pg_engine)
    log.info("  %s rows", f"{len(df):,}")
    df.to_sql(target_table, sqlite_engine, if_exists="append", index=False)


def main() -> None:
    import os
    if not os.getenv("USE_POSTGRES", "").lower() == "true":
        log.error("Set USE_POSTGRES=true before running this script.")
        sys.exit(1)

    pg_engine = create_engine(pg_url(), echo=False)

    # Remove old SQLite DB so we start fresh
    if SQLITE_DB_PATH.exists():
        SQLITE_DB_PATH.unlink()
        log.info("Removed existing %s", SQLITE_DB_PATH)

    sqlite_engine = create_engine(sqlite_url(), echo=False)
    setup_sqlite(sqlite_engine)

    # ── Base tables ───────────────────────────────────────────────────────────
    export_table(pg_engine, sqlite_engine, "customers")
    export_table(pg_engine, sqlite_engine, "products")
    export_table(pg_engine, sqlite_engine, "invoices")
    # invoice_items: drop 'line_total' and 'id' — both are generated/serial in SQLite
    export_table(pg_engine, sqlite_engine, "invoice_items",
                 drop_cols=["id", "line_total"])
    export_table(pg_engine, sqlite_engine, "customer_segments")

    # ── Materialised views ────────────────────────────────────────────────────
    export_view(pg_engine, sqlite_engine, "vw_revenue_by_country", "revenue_by_country")

    log.info("SQLite export complete → %s", SQLITE_DB_PATH)

    # Quick row counts
    with sqlite_engine.connect() as conn:
        for tbl in ["customers", "products", "invoices", "invoice_items",
                    "customer_segments", "revenue_by_country"]:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
            log.info("  %-25s %s rows", tbl, f"{n:,}")


if __name__ == "__main__":
    main()
