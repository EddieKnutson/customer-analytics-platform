# Customer Analytics & Experimentation Platform

A full-stack analytics portfolio project built on the UCI Online Retail dataset
(~500K transactions, UK-based e-commerce retailer, 2010–2011).

**[Live Demo →](https://customer-analytics-platform-cmcent8tnrbscecdizu42i.streamlit.app)**

---

## What it does

| Module | Description |
|--------|-------------|
| **ETL pipeline** | Loads UCI Online Retail `.xlsx` → PostgreSQL (local) / SQLite (deployed) with full data cleaning |
| **RFM segmentation** | K-Means + hierarchical clustering on recency/frequency/monetary quintiles → 5 labeled customer segments |
| **Cohort retention** | Month-of-first-purchase cohort × months-since-join retention matrix |
| **A/B testing** | Two-proportion z-test, chi-squared, Bayesian Beta-Binomial, power analysis |
| **Streamlit dashboard** | 4-page interactive app: KPIs, RFM scatter/radar, cohort heatmap, experiment calculator |

---

## Architecture

```
customer-analytics-platform/
├── config.py                   # DB config, paths, settings
├── data/
│   ├── online_retail.xlsx      # Raw dataset (download separately)
│   └── retail.db               # SQLite export for deployment
├── sql/
│   ├── schema.sql              # PostgreSQL DDL
│   ├── vw_rfm_scores.sql       # Raw RFM metrics per customer
│   ├── vw_rfm_quintiles.sql    # NTILE(5) scoring
│   ├── vw_cohort_retention.sql # Cohort × retention matrix
│   └── vw_revenue_by_country.sql
├── etl/
│   ├── load_data.py            # XLSX → PostgreSQL / SQLite
│   └── export_sqlite.py        # PostgreSQL → SQLite for deployment
├── analytics/
│   ├── clustering.py           # K-Means + hierarchical, elbow method
│   └── ab_testing.py           # z-test, chi-squared, Bayesian, power analysis
└── app/
    ├── main.py                 # Streamlit entry point
    ├── db.py                   # Shared DB connection + query helpers
    └── pages/
        ├── 1_Overview.py
        ├── 2_RFM_Segmentation.py
        ├── 3_Cohort_Retention.py
        └── 4_Experimentation.py
```

**Data flow:**
```
online_retail.xlsx
       │
  etl/load_data.py  ──►  PostgreSQL (local dev)
       │                        │
       │              sql/schema.sql + views
       │                        │
       │              analytics/clustering.py  ──►  customer_segments table
       │                        │
       └──►  SQLite (deploy)  ◄──  etl/export_sqlite.py
                    │
              Streamlit Cloud
```

---

## SQL Showcase

**RFM quintile scoring** — window functions, NTILE scoring, CTE chaining:

```sql
CREATE OR REPLACE VIEW vw_rfm_quintiles AS
WITH quintiles AS (
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,  -- lower days = better
        NTILE(5) OVER (ORDER BY frequency ASC)     AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC)      AS m_score
    FROM vw_rfm_scores
)
SELECT *, (r_score + f_score + m_score) AS rfm_score
FROM quintiles;
```

**Cohort retention** — self-join on first-purchase month, window functions:

```sql
WITH customer_cohorts AS (
    SELECT customer_id,
           DATE_TRUNC('month', MIN(invoice_date)) AS cohort_month
    FROM invoices WHERE is_cancelled = FALSE
    GROUP BY customer_id
),
cohort_data AS (
    SELECT cc.cohort_month, ca.activity_month,
           EXTRACT(YEAR  FROM AGE(ca.activity_month, cc.cohort_month))::INT * 12
           + EXTRACT(MONTH FROM AGE(ca.activity_month, cc.cohort_month))::INT
               AS months_since_join,
           cc.customer_id
    FROM customer_cohorts cc
    JOIN customer_activity ca ON ca.customer_id = cc.customer_id
)
SELECT cohort_month, months_since_join,
       COUNT(DISTINCT customer_id) AS customers,
       ROUND(COUNT(DISTINCT customer_id)::NUMERIC / cohort_size, 4) AS retention_rate
FROM cohort_data JOIN cohort_sizes USING (cohort_month)
GROUP BY cohort_month, months_since_join, cohort_size;
```

---

## Setup

**1. Clone and install dependencies**
```bash
git clone <repo>
cd customer-analytics-platform
pip install -r requirements.txt
```

**2. Download the dataset**

Download `online_retail.xlsx` from [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
and place it at `data/online_retail.xlsx`.

**3. Set up PostgreSQL** *(local dev)*
```bash
createdb -U postgres -h localhost retail_analytics
psql -U postgres -h localhost -d retail_analytics -f sql/schema.sql
psql -U postgres -h localhost -d retail_analytics -f sql/vw_rfm_scores.sql
psql -U postgres -h localhost -d retail_analytics -f sql/vw_rfm_quintiles.sql
psql -U postgres -h localhost -d retail_analytics -f sql/vw_cohort_retention.sql
psql -U postgres -h localhost -d retail_analytics -f sql/vw_revenue_by_country.sql
```

**4. Run the ETL**
```bash
USE_POSTGRES=true PG_USER=postgres PG_PASSWORD=... python -m etl.load_data
```

**5. Run clustering**
```bash
USE_POSTGRES=true PG_USER=postgres PG_PASSWORD=... python -m analytics.clustering --k 5
```

**6. Launch the dashboard**
```bash
USE_POSTGRES=true PG_USER=postgres PG_PASSWORD=... streamlit run app/main.py
```

---

## Deploying to Streamlit Cloud

**1. Export to SQLite**
```bash
USE_POSTGRES=true PG_USER=postgres PG_PASSWORD=... python -m etl.export_sqlite
```

**2. Commit `data/retail.db` to the repo** *(or upload via Streamlit Cloud secrets)*

**3. Push and connect repo to [share.streamlit.io](https://share.streamlit.io)**
- Main file: `app/main.py`
- No environment variables needed (SQLite is the default backend)

---

## Tech Stack

`Python 3.12` · `PostgreSQL 17` · `SQLite` · `SQLAlchemy` · `Pandas` · `Scikit-learn` · `SciPy` · `Streamlit` · `Plotly`

---

*Dataset: Daqing Chen et al., "Data Mining for the Online Retail Industry", UCI ML Repository, 2015.*
