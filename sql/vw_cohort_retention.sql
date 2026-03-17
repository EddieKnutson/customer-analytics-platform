-- vw_cohort_retention.sql — Monthly cohort × retention matrix
--
-- A "cohort" = the calendar month of a customer's first-ever purchase.
-- For each cohort we track how many of those customers returned in
-- month 0 (acquisition month), month 1, month 2, … up to month 12.
--
-- Output columns:
--   cohort_month      — YYYY-MM-01 truncated date of first purchase
--   months_since_join — 0, 1, 2, … (integer months after cohort month)
--   customers         — distinct customers active in that cohort × period cell
--   cohort_size       — total customers in the cohort (for % calculation)
--   retention_rate    — customers / cohort_size (0.0 – 1.0)
--
-- Used by app/pages/3_Cohort_Retention.py to render the heatmap.

CREATE OR REPLACE VIEW vw_cohort_retention AS
WITH
-- Step 1: find each customer's cohort month (their very first purchase month)
customer_cohorts AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', MIN(invoice_date)) AS cohort_month
    FROM invoices
    WHERE is_cancelled = FALSE
    GROUP BY customer_id
),

-- Step 2: all active months per customer (months they actually purchased)
customer_activity AS (
    SELECT DISTINCT
        i.customer_id,
        DATE_TRUNC('month', i.invoice_date) AS activity_month
    FROM invoices i
    WHERE i.is_cancelled = FALSE
),

-- Step 3: join activity back to cohort to get months_since_join
cohort_data AS (
    SELECT
        cc.cohort_month,
        ca.activity_month,
        -- Integer month difference using EXTRACT on the interval
        EXTRACT(
            YEAR  FROM AGE(ca.activity_month, cc.cohort_month)
        )::INT * 12
        + EXTRACT(
            MONTH FROM AGE(ca.activity_month, cc.cohort_month)
        )::INT                                  AS months_since_join,
        cc.customer_id
    FROM customer_cohorts  cc
    JOIN customer_activity ca ON ca.customer_id = cc.customer_id
    -- Only forward in time (activity_month >= cohort_month is guaranteed
    -- since cohort_month is the minimum, but explicit guard is safer)
    WHERE ca.activity_month >= cc.cohort_month
),

-- Step 4: cohort sizes (denominator for retention rate)
cohort_sizes AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT customer_id) AS cohort_size
    FROM customer_cohorts
    GROUP BY cohort_month
)

SELECT
    cd.cohort_month,
    cd.months_since_join,
    COUNT(DISTINCT cd.customer_id)              AS customers,
    cs.cohort_size,
    ROUND(
        COUNT(DISTINCT cd.customer_id)::NUMERIC
        / cs.cohort_size,
        4
    )                                           AS retention_rate
FROM cohort_data   cd
JOIN cohort_sizes  cs ON cs.cohort_month = cd.cohort_month
GROUP BY
    cd.cohort_month,
    cd.months_since_join,
    cs.cohort_size
ORDER BY
    cd.cohort_month,
    cd.months_since_join;
