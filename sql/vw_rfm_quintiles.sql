-- vw_rfm_quintiles.sql — NTILE(5) quintile scoring on raw RFM metrics
--
-- Builds on vw_rfm_scores (recency_days, frequency, monetary).
--
-- Scoring logic:
--   R score: HIGHER is better → fewer recency_days = score 5
--            Use NTILE ordered DESC so smallest recency_days lands in bucket 5
--   F score: HIGHER is better → more purchases = score 5
--            Use NTILE ordered ASC
--   M score: HIGHER is better → more spend = score 5
--            Use NTILE ordered ASC
--
-- rfm_score (3–15) = r_score + f_score + m_score
-- Used downstream by analytics/clustering.py as input features.

CREATE OR REPLACE VIEW vw_rfm_quintiles AS
WITH quintiles AS (
    SELECT
        customer_id,
        country,
        recency_days,
        frequency,
        monetary,
        last_purchase_date,

        -- Recency: order DESC so the most-recent customers (smallest days) get score 5
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,

        -- Frequency: order ASC so highest purchase count gets score 5
        NTILE(5) OVER (ORDER BY frequency ASC)     AS f_score,

        -- Monetary: order ASC so highest spenders get score 5
        NTILE(5) OVER (ORDER BY monetary ASC)      AS m_score

    FROM vw_rfm_scores
)
SELECT
    customer_id,
    country,
    recency_days,
    frequency,
    ROUND(monetary::NUMERIC, 2) AS monetary,
    last_purchase_date,
    r_score,
    f_score,
    m_score,
    (r_score + f_score + m_score) AS rfm_score
FROM quintiles;
