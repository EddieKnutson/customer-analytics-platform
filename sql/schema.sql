-- schema.sql — PostgreSQL schema for UCI Online Retail dataset
-- Run once to set up tables before ETL

-- Drop in reverse dependency order if rebuilding
DROP TABLE IF EXISTS invoice_items;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

-- ── Core tables ────────────────────────────────────────────────────────────────

CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    country       VARCHAR(100) NOT NULL
);

CREATE TABLE products (
    stock_code    VARCHAR(20)  PRIMARY KEY,
    description   TEXT
);

CREATE TABLE invoices (
    invoice_no    VARCHAR(20)  PRIMARY KEY,
    customer_id   INTEGER      REFERENCES customers(customer_id),
    invoice_date  TIMESTAMP    NOT NULL,
    is_cancelled  BOOLEAN      NOT NULL DEFAULT FALSE
);

CREATE TABLE invoice_items (
    id            SERIAL       PRIMARY KEY,
    invoice_no    VARCHAR(20)  REFERENCES invoices(invoice_no),
    stock_code    VARCHAR(20)  REFERENCES products(stock_code),
    quantity      INTEGER      NOT NULL,
    unit_price    NUMERIC(10,2) NOT NULL,
    line_total    NUMERIC(10,2) GENERATED ALWAYS AS (quantity * unit_price) STORED
);

-- ── Indexes ────────────────────────────────────────────────────────────────────
CREATE INDEX idx_invoices_customer   ON invoices(customer_id);
CREATE INDEX idx_invoices_date       ON invoices(invoice_date);
CREATE INDEX idx_items_invoice       ON invoice_items(invoice_no);
CREATE INDEX idx_items_stock         ON invoice_items(stock_code);
