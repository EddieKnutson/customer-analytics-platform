"""
analytics/clustering.py — RFM Customer Segmentation

Pipeline:
  1. Load vw_rfm_quintiles from DB
  2. Scale R/F/M scores (StandardScaler)
  3. Elbow method → suggest best k (also saves elbow plot)
  4. K-Means clustering at chosen k
  5. Hierarchical clustering (Ward linkage) as cross-check
  6. Assign human-readable segment labels based on mean R/F/M per cluster
  7. Save results to `customer_segments` table

Usage:
    # Run with elbow plot, then re-run with chosen k:
    python -m analytics.clustering                  # uses N_CLUSTERS from config
    python -m analytics.clustering --k 4            # override k
    python -m analytics.clustering --elbow-only     # just plot elbow, don't cluster
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server/CI
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import db_url, N_CLUSTERS, RANDOM_STATE, BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PLOTS_DIR = BASE_DIR / "output" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Features used for clustering — the three normalised quintile scores
FEATURES = ["r_score", "f_score", "m_score"]

# ── Segment label assignment ──────────────────────────────────────────────────
# After clustering we rank each cluster by its mean RFM scores and assign a
# business label.  The logic is intentionally simple: Champions have high R+F+M,
# Lapsed have low R and low F, etc.  Labels are derived programmatically so
# they're reproducible regardless of which cluster ID lands where.

def assign_segment_label(row: pd.Series) -> str:
    """
    Rule-based label derived from a cluster's mean r/f/m quintile scores (1–5).
    Called on the cluster-level summary DataFrame, one row per cluster.
    """
    r, f, m = row["r_score"], row["f_score"], row["m_score"]
    rfm = r + f + m  # composite 3–15

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "New"
    if r <= 2 and f >= 3 and m >= 3:
        return "At-Risk"
    if r <= 2 and f <= 2:
        return "Lapsed"
    return "Potential"  # catch-all for mid-tier customers


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_rfm(engine) -> pd.DataFrame:
    log.info("Loading vw_rfm_quintiles …")
    df = pd.read_sql("SELECT * FROM vw_rfm_quintiles", engine)
    log.info("Loaded %s customers", f"{len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. ELBOW METHOD
# ─────────────────────────────────────────────────────────────────────────────

def elbow_analysis(X_scaled: np.ndarray, k_range=range(2, 11)) -> dict:
    """
    Compute inertia and silhouette score for each k.
    Returns dict with keys 'k', 'inertia', 'silhouette'.
    """
    results = {"k": [], "inertia": [], "silhouette": []}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(X_scaled)
        results["k"].append(k)
        results["inertia"].append(km.inertia_)
        results["silhouette"].append(silhouette_score(X_scaled, labels))
        log.info("  k=%d  inertia=%.1f  silhouette=%.4f", k, km.inertia_, results["silhouette"][-1])
    return results


def plot_elbow(results: dict) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(results["k"], results["inertia"], marker="o", color="#2563EB")
    ax1.set_title("Elbow — Inertia vs k")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster SSE)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(results["k"], results["silhouette"], marker="o", color="#16A34A")
    best_k = results["k"][int(np.argmax(results["silhouette"]))]
    ax2.axvline(best_k, linestyle="--", color="#DC2626", alpha=0.7,
                label=f"Best silhouette: k={best_k}")
    ax2.set_title("Silhouette Score vs k")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette score (higher = better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "elbow_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Elbow plot saved → %s", path)
    return path


def suggest_k(results: dict) -> int:
    """Pick k with the highest silhouette score."""
    best_idx = int(np.argmax(results["silhouette"]))
    return results["k"][best_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(X_scaled: np.ndarray, k: int) -> np.ndarray:
    log.info("Running K-Means with k=%d …", k)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    log.info("K-Means silhouette score: %.4f", sil)
    return labels


def run_hierarchical(X_scaled: np.ndarray, k: int) -> np.ndarray:
    """
    Ward linkage minimises within-cluster variance — closest in spirit to
    K-Means but doesn't require choosing k up front (we pass k here as
    a cross-check, not as the primary method).
    """
    log.info("Running Hierarchical clustering (Ward, k=%d) as cross-check …", k)
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = hc.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    log.info("Hierarchical silhouette score: %.4f", sil)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEGMENT LABELLING
# ─────────────────────────────────────────────────────────────────────────────

def label_clusters(df: pd.DataFrame, labels: np.ndarray, method: str = "kmeans") -> pd.DataFrame:
    df = df.copy()
    df[f"{method}_cluster"] = labels

    # Mean scores per cluster → derive label
    cluster_means = (
        df.groupby(f"{method}_cluster")[FEATURES]
        .mean()
        .reset_index()
    )
    cluster_means.columns = [f"{method}_cluster", "r_score", "f_score", "m_score"]
    cluster_means["segment"] = cluster_means.apply(assign_segment_label, axis=1)

    log.info("Cluster → segment mapping:")
    for _, row in cluster_means.iterrows():
        log.info("  cluster %d → %-12s  (R=%.2f F=%.2f M=%.2f)",
                 row[f"{method}_cluster"], row["segment"],
                 row["r_score"], row["f_score"], row["m_score"])

    df = df.merge(
        cluster_means[[f"{method}_cluster", "segment"]],
        on=f"{method}_cluster",
        how="left",
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. PERSIST RESULTS
# ─────────────────────────────────────────────────────────────────────────────

CREATE_SEGMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS customer_segments (
    customer_id     INTEGER PRIMARY KEY,
    country         VARCHAR(100),
    recency_days    INTEGER,
    frequency       INTEGER,
    monetary        NUMERIC(12,2),
    r_score         INTEGER,
    f_score         INTEGER,
    m_score         INTEGER,
    rfm_score       INTEGER,
    kmeans_cluster  INTEGER,
    hc_cluster      INTEGER,
    segment         VARCHAR(50)
);
"""

# For re-runs we truncate and reload
TRUNCATE_SEGMENTS = "DELETE FROM customer_segments;"


def save_segments(df: pd.DataFrame, engine) -> None:
    cols = [
        "customer_id", "country", "recency_days", "frequency", "monetary",
        "r_score", "f_score", "m_score", "rfm_score",
        "kmeans_cluster", "hc_cluster", "segment",
    ]
    # Only keep columns that exist (hc_cluster may be absent if skipped)
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    with engine.begin() as conn:
        conn.execute(text(CREATE_SEGMENTS_TABLE))
        conn.execute(text(TRUNCATE_SEGMENTS))

    log.info("Saving %s rows to customer_segments …", f"{len(out):,}")
    out.to_sql("customer_segments", engine, if_exists="append", index=False)
    log.info("customer_segments saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SEGMENT SUMMARY PRINTOUT
# ─────────────────────────────────────────────────────────────────────────────

def print_segment_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .sort_values("customers", ascending=False)
        .reset_index()
    )
    summary["avg_recency"]   = summary["avg_recency"].round(0).astype(int)
    summary["avg_frequency"] = summary["avg_frequency"].round(1)
    summary["avg_monetary"]  = summary["avg_monetary"].round(0).astype(int)

    log.info("─── Segment summary ─────────────────────────────────────────")
    log.info("  %-12s  %7s  %10s  %12s  %12s",
             "Segment", "Cust", "Avg Recency", "Avg Freq", "Avg Spend £")
    for _, row in summary.iterrows():
        log.info("  %-12s  %7s  %10s  %12s  %12s",
                 row["segment"],
                 f"{row['customers']:,}",
                 f"{row['avg_recency']}d",
                 f"{row['avg_frequency']}",
                 f"£{row['avg_monetary']:,}")
    log.info("─────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(k: int = None, elbow_only: bool = False) -> None:
    engine = create_engine(db_url(), echo=False)

    df = load_rfm(engine)

    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow analysis — always run so the plot is available
    log.info("Running elbow analysis (k=2 to 10) …")
    elbow_results = elbow_analysis(X_scaled)
    plot_elbow(elbow_results)
    suggested_k = suggest_k(elbow_results)
    log.info("Suggested k (best silhouette): %d", suggested_k)

    if elbow_only:
        log.info("--elbow-only flag set. Stopping here.")
        log.info("Re-run without --elbow-only (or with --k %d) to cluster.", suggested_k)
        return

    # Use provided k, or config default, or silhouette-suggested k
    if k is None:
        k = N_CLUSTERS if N_CLUSTERS != suggested_k else suggested_k
        log.info("Using k=%d (from config/suggestion)", k)

    # K-Means
    km_labels = run_kmeans(X_scaled, k)
    df = label_clusters(df, km_labels, method="kmeans")

    # Hierarchical (cross-check — uses same k)
    hc_labels = run_hierarchical(X_scaled, k)
    df["hc_cluster"] = hc_labels

    # Print summary
    print_segment_summary(df)

    # Persist
    save_segments(df, engine)

    log.info("Clustering complete. Results in customer_segments table.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFM Customer Clustering")
    parser.add_argument("--k",           type=int,  default=None, help="Number of clusters (overrides config)")
    parser.add_argument("--elbow-only",  action="store_true",     help="Only run elbow analysis, skip clustering")
    args = parser.parse_args()
    main(k=args.k, elbow_only=args.elbow_only)
