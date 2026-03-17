"""
etl/load_data.py — UCI Online Retail dataset → PostgreSQL / SQLite

Usage:
    python -m etl.load_data                         # SQLite (default)
    USE_POSTGRES=true python -m etl.load_data       # PostgreSQL (run sql/schema.sql first)

Cleaning rules applied:
- Rows with null CustomerID      → dropped (can't associate with a customer)
- Cancelled invoices (InvoiceNo starts with 'C') → loaded with is_cancelled=True
- Negative quantity / zero-or-negative unit_price → kept in DB but flagged;
  the RFM views already filter these out with quantity > 0 AND unit_price > 0
- Duplicate StockCode descriptions → first-seen description wins
- Duplicate CustomerID × Country  → first-seen country wins
"""

import sys
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

# ── Make sure the project root is on sys.path when run as a script ───────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_PATH, db_url, USE_POSTGRES

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── SQLite DDL (used when USE_POSTGRES is False) ──────────────────────────────
# Mirrors schema.sql but uses SQLite-compatible syntax.
# line_total is a stored generated column (SQLite ≥ 3.31; bundled in Python 3.8+).
SQLITE_DDL = """
DROP TABLE IF EXISTS invoice_items;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

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
    is_cancelled  BOOLEAN NOT NULL DEFAULT 0
);

CREATE TABLE invoice_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_no    VARCHAR(20) REFERENCES invoices(invoice_no),
    stock_code    VARCHAR(20) REFERENCES products(stock_code),
    quantity      INTEGER NOT NULL,
    unit_price    REAL NOT NULL,
    line_total    REAL GENERATED ALWAYS AS (quantity * unit_price) STORED
);

CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices(customer_id);
CREATE INDEX IF NOT EXISTS idx_invoices_date     ON invoices(invoice_date);
CREATE INDEX IF NOT EXISTS idx_items_invoice     ON invoice_items(invoice_no);
CREATE INDEX IF NOT EXISTS idx_items_stock       ON invoice_items(stock_code);
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    log.info("Reading %s …", path)
    df = pd.read_excel(path, engine="openpyxl", dtype={"CustomerID": "Int64"})
    log.info("Raw rows: %s", f"{len(df):,}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # ── Standardise column names ──────────────────────────────────────────────
    df.columns = [c.strip() for c in df.columns]

    # ── Drop rows without a CustomerID ───────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["CustomerID"])
    log.info("Dropped %s rows with null CustomerID", f"{before - len(df):,}")

    # ── Cast types ────────────────────────────────────────────────────────────
    df["CustomerID"]  = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Quantity"]    = df["Quantity"].astype(int)
    df["UnitPrice"]   = df["UnitPrice"].astype(float)

    # ── Cancelled flag ────────────────────────────────────────────────────────
    df["is_cancelled"] = df["InvoiceNo"].astype(str).str.startswith("C")

    # ── Strip whitespace from string columns ──────────────────────────────────
    df["InvoiceNo"]   = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"]   = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    df["Country"]     = df["Country"].astype(str).str.strip()

    log.info("Clean rows: %s  (cancelled: %s)",
             f"{len(df):,}",
             f"{df['is_cancelled'].sum():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD DIMENSION / FACT DATAFRAMES
# ─────────────────────────────────────────────────────────────────────────────

def build_customers(df: pd.DataFrame) -> pd.DataFrame:
    """One row per CustomerID — first-seen Country wins."""
    return (
        df[["CustomerID", "Country"]]
        .drop_duplicates(subset="CustomerID", keep="first")
        .rename(columns={"CustomerID": "customer_id", "Country": "country"})
        .reset_index(drop=True)
    )


def build_products(df: pd.DataFrame) -> pd.DataFrame:
    """One row per StockCode — first non-empty Description wins."""
    products = (
        df[["StockCode", "Description"]]
        .copy()
    )
    # Prefer non-empty description
    products = products.sort_values(
        "Description",
        key=lambda s: s.str.len(),
        ascending=False,
    )
    products = (
        products
        .drop_duplicates(subset="StockCode", keep="first")
        .rename(columns={"StockCode": "stock_code", "Description": "description"})
        .reset_index(drop=True)
    )
    products["description"] = products["description"].replace("", None)
    return products


def build_invoices(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["InvoiceNo", "CustomerID", "InvoiceDate", "is_cancelled"]]
        .drop_duplicates(subset="InvoiceNo", keep="first")
        .rename(columns={
            "InvoiceNo":    "invoice_no",
            "CustomerID":   "customer_id",
            "InvoiceDate":  "invoice_date",
        })
        .reset_index(drop=True)
    )


def build_invoice_items(df: pd.DataFrame) -> pd.DataFrame:
    items = df[["InvoiceNo", "StockCode", "Quantity", "UnitPrice"]].copy()
    items = items.rename(columns={
        "InvoiceNo":  "invoice_no",
        "StockCode":  "stock_code",
        "Quantity":   "quantity",
        "UnitPrice":  "unit_price",
    })
    return items.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATABASE SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_sqlite(engine) -> None:
    """Create tables in SQLite (idempotent — drops and recreates)."""
    with engine.begin() as conn:
        for stmt in SQLITE_DDL.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    log.info("SQLite schema created.")


def setup_postgres(engine) -> None:
    """
    For PostgreSQL we expect schema.sql was already applied.
    We just truncate all tables so re-runs are idempotent.
    """
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE invoice_items, invoices, products, customers RESTART IDENTITY CASCADE"))
    log.info("PostgreSQL tables truncated — ready for load.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD
# ─────────────────────────────────────────────────────────────────────────────

CHUNK = 10_000  # rows per batch for invoice_items


def load_tables(engine, customers, products, invoices, items) -> None:
    is_pg = USE_POSTGRES

    # ── customers ────────────────────────────────────────────────────────────
    log.info("Loading customers (%s rows) …", f"{len(customers):,}")
    customers.to_sql("customers", engine, if_exists="append", index=False)

    # ── products ─────────────────────────────────────────────────────────────
    log.info("Loading products (%s rows) …", f"{len(products):,}")
    products.to_sql("products", engine, if_exists="append", index=False)

    # ── invoices ─────────────────────────────────────────────────────────────
    log.info("Loading invoices (%s rows) …", f"{len(invoices):,}")
    invoices.to_sql("invoices", engine, if_exists="append", index=False)

    # ── invoice_items ─────────────────────────────────────────────────────────
    # PostgreSQL: line_total is GENERATED ALWAYS AS — must NOT be in the INSERT
    # SQLite:     line_total is also GENERATED ALWAYS AS — same rule applies
    # pandas to_sql does not insert the 'id' (autoincrement PK) either since
    # it's not in our DataFrame.
    log.info("Loading invoice_items (%s rows) in chunks of %s …",
             f"{len(items):,}", f"{CHUNK:,}")

    total_chunks = (len(items) // CHUNK) + 1
    for i, chunk_start in enumerate(range(0, len(items), CHUNK), 1):
        chunk = items.iloc[chunk_start: chunk_start + CHUNK]
        chunk.to_sql("invoice_items", engine, if_exists="append", index=False)
        if i % 10 == 0 or i == total_chunks:
            log.info("  … chunk %s / %s", i, total_chunks)

    log.info("invoice_items load complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(engine) -> None:
    queries = {
        "customers":     "SELECT COUNT(*) FROM customers",
        "products":      "SELECT COUNT(*) FROM products",
        "invoices":      "SELECT COUNT(*) FROM invoices",
        "invoice_items": "SELECT COUNT(*) FROM invoice_items",
        "cancelled":     "SELECT COUNT(*) FROM invoices WHERE is_cancelled = TRUE" if USE_POSTGRES
                         else "SELECT COUNT(*) FROM invoices WHERE is_cancelled = 1",
        "date_range":    "SELECT MIN(invoice_date), MAX(invoice_date) FROM invoices",
    }
    with engine.connect() as conn:
        log.info("─── Load summary ────────────────────────────────────")
        for label, q in queries.items():
            result = conn.execute(text(q)).fetchone()
            log.info("  %-18s %s", label + ":", "  |  ".join(str(v) for v in result))
        log.info("─────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not RAW_DATA_PATH.exists():
        log.error("Dataset not found at %s", RAW_DATA_PATH)
        log.error("Download from https://archive.ics.uci.edu/dataset/352/online+retail")
        log.error("and place the .xlsx file at data/online_retail.xlsx")
        sys.exit(1)

    # 1. Load & clean
    df = load_raw(RAW_DATA_PATH)
    df = clean(df)

    # 2. Build dimension / fact tables
    customers = build_customers(df)
    products  = build_products(df)
    invoices  = build_invoices(df)
    items     = build_invoice_items(df)

    log.info("Dimensions — customers: %s | products: %s | invoices: %s | items: %s",
             f"{len(customers):,}", f"{len(products):,}",
             f"{len(invoices):,}", f"{len(items):,}")

    # 3. Connect & set up DB
    url = db_url()
    log.info("Connecting to: %s", url.split("@")[-1] if "@" in url else url)
    engine = create_engine(url, echo=False)

    if USE_POSTGRES:
        setup_postgres(engine)
    else:
        setup_sqlite(engine)

    # 4. Load
    load_tables(engine, customers, products, invoices, items)

    # 5. Summary
    print_summary(engine)

    log.info("ETL complete.")


if __name__ == "__main__":
    main()
