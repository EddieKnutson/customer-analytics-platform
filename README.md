# Customer Analytics Platform

End-to-end customer analytics on the UCI Online Retail dataset (~500K transactions from a UK-based e-commerce platform). The project covers EDA, data cleaning, RFM feature engineering, K-Means customer segmentation, and a Tableau dashboard for interactive exploration.

**Author:** Eddie Knutson — Data Analytics, Washington State University (May 2026)

---

## Tableau Dashboard

**[View the interactive dashboard on Tableau Public →](https://public.tableau.com/app/profile/eddie.knutson/viz/Customer-Analytics_17784539951960/Dashboard1)**

The dashboard visualizes customer spending patterns, top markets, price-tier distribution, and the K-Means cluster segments produced by the notebook.

---

## What's in this repo

| File | Description |
| --- | --- |
| `EDA.ipynb` | Full analysis notebook: cleaning, EDA, RFM features, clustering, and CSV export for Tableau |
| `README.md` | This file |
| `.gitignore` | Excludes raw/processed data files (kept locally — see below) |

### Data files (not tracked in git)

These are excluded because of size; regenerate them by running the notebook against the source dataset:

- `online_retail.xlsx` — raw source data ([UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail))
- `retail_transactions_final.csv` — cleaned transaction-level data, exported for Tableau
- `customer_segments_final.csv` — RFM features + cluster assignments per customer

---

## Analysis Workflow

1. **Cleaning** — Drop rows with missing `CustomerID`, remove negative quantities (returns), filter out $0 / $0.01 line items (bank charges, misc.).
2. **EDA** — Monthly quantity distributions, top/bottom 10 customers by spend, top 10 countries by revenue, unit-price distribution (log-scaled).
3. **Price tiering** — `pd.qcut` splits items into Low / Medium / High price tiers.
4. **RFM features** — Per-customer Recency (days since last purchase), Frequency (unique orders), Monetary (total spend), plus total quantity.
5. **Clustering** — Log-transform + standard-scale RFM features, choose `k=3` via elbow method, fit K-Means.
6. **Export** — Cleaned transactions and customer segments saved as CSV for the Tableau dashboard.

---

## Key Findings

- Revenue is heavily UK-concentrated (~7M EUR vs. <1M EUR for any other country).
- Customer spend is highly skewed — the top customer spent ~77K EUR while many spent under 1 EUR.
- Three customer segments emerge cleanly from RFM: high-value loyal, mid-tier regular, and low-engagement / at-risk.

---

## Tools

`Python` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn` · `Jupyter` · `Tableau`
