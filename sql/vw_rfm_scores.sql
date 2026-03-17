-- vw_rfm_scores.sql — Raw RFM metrics per customer (not yet scored)
-- Recency  = days since last purchase
-- Frequency = distinct invoice count
-- Monetary  = total spend

CREATE OR REPLACE VIEW vw_rfm_scores AS
WITH snapshot AS (
    -- Use max invoice date + 1 as "today" so dataset is self-contained
    SELECT MAX(invoice_date) + INTERVAL '1 day' AS snapshot_date
    FROM invoices
    WHERE is_cancelled = FALSE
),
customer_spend AS (
    SELECT
        i.customer_id,
        MAX(i.invoice_date)                    AS last_purchase_date,
        COUNT(DISTINCT i.invoice_no)           AS frequency,
        SUM(ii.line_total)                     AS monetary
    FROM invoices i
    JOIN invoice_items ii ON ii.invoice_no = i.invoice_no
    WHERE i.is_cancelled = FALSE
      AND ii.quantity > 0
      AND ii.unit_price > 0
    GROUP BY i.customer_id
)
SELECT
    cs.customer_id,
    c.country,
    EXTRACT(DAY FROM (s.snapshot_date - cs.last_purchase_date))::INT AS recency_days,
    cs.frequency,
    ROUND(cs.monetary::NUMERIC, 2) AS monetary,
    cs.last_purchase_date
FROM customer_spend cs
JOIN customers c  ON c.customer_id = cs.customer_id
CROSS JOIN snapshot s;
