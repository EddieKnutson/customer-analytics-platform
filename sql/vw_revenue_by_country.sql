-- vw_revenue_by_country.sql — Revenue + customer count by country
--
-- Filters:
--   - Excludes cancelled invoices (is_cancelled = FALSE)
--   - Excludes negative quantities and zero/negative prices
--     (same rule as vw_rfm_scores — keeps analysis consistent)
--
-- Output columns:
--   country           — customer country
--   total_revenue     — sum of line_total across all valid items
--   customer_count    — distinct customers who purchased from that country
--   order_count       — distinct invoices (orders)
--   avg_order_value   — total_revenue / order_count
--
-- Used by app/pages/1_Overview.py for the Plotly choropleth map.

CREATE OR REPLACE VIEW vw_revenue_by_country AS
SELECT
    c.country,
    ROUND(SUM(ii.line_total)::NUMERIC,  2)  AS total_revenue,
    COUNT(DISTINCT i.customer_id)            AS customer_count,
    COUNT(DISTINCT i.invoice_no)             AS order_count,
    ROUND(
        SUM(ii.line_total)::NUMERIC
        / NULLIF(COUNT(DISTINCT i.invoice_no), 0),
        2
    )                                        AS avg_order_value
FROM invoices      i
JOIN invoice_items ii ON ii.invoice_no  = i.invoice_no
JOIN customers     c  ON c.customer_id  = i.customer_id
WHERE i.is_cancelled   = FALSE
  AND ii.quantity      > 0
  AND ii.unit_price    > 0
GROUP BY c.country
ORDER BY total_revenue DESC;
