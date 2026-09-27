import pandas as pd

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_PATH = Path(
    "data/processed/clustering_features.csv"
)

METRICS_PATH = Path(
    "data/processed/customer_metrics.csv"
)

RANDOM_STATE = 42
N_CLUSTERS = 3


# ============================================================
# LOAD DATA
# ============================================================

features = pd.read_csv(FEATURES_PATH)

customer_metrics = pd.read_csv(
    METRICS_PATH
)

print(
    f"Loaded {len(features):,} clustering records"
)


# ============================================================
# DEFINE FEATURE SETS
# ============================================================

features_with_returns = [
    "scaled_log_total_orders",
    "scaled_log_total_revenue",
    "scaled_log_average_order_value",
    "scaled_log_recency_days",
    "scaled_category_count",
    "scaled_average_discount",
    "scaled_return_rate"
]

features_without_returns = [
    "scaled_log_total_orders",
    "scaled_log_total_revenue",
    "scaled_log_average_order_value",
    "scaled_log_recency_days",
    "scaled_category_count",
    "scaled_average_discount"
]


# ============================================================
# FUNCTION TO EVALUATE MODEL
# ============================================================

def evaluate_model(
    feature_set,
    model_name
):

    X = features[feature_set]

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = model.fit_predict(X)

    silhouette = silhouette_score(
        X,
        labels
    )

    inertia = model.inertia_

    assignments = pd.DataFrame({
        "customer_id": features["customer_id"],
        "cluster": labels
    })

    segmented = customer_metrics.merge(
        assignments,
        on="customer_id",
        how="inner"
    )

    # --------------------------------------------------------
    # Cluster sizes
    # --------------------------------------------------------

    cluster_sizes = (
        segmented["cluster"]
        .value_counts()
        .sort_index()
    )

    cluster_pct = (
        segmented["cluster"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    # --------------------------------------------------------
    # Business profiles
    # --------------------------------------------------------

    profile_columns = [
        "total_orders",
        "total_revenue",
        "average_order_value",
        "recency_days",
        "category_count",
        "average_discount",
        "return_rate"
    ]

    profiles = (
        segmented
        .groupby("cluster")[profile_columns]
        .mean()
        .round(2)
    )

    # --------------------------------------------------------
    # Revenue contribution
    # --------------------------------------------------------

    revenue = (
        segmented
        .groupby("cluster")["total_revenue"]
        .sum()
    )

    revenue_pct = (
        revenue /
        segmented["total_revenue"].sum()
        * 100
    )

    # --------------------------------------------------------
    # Recency behavior
    # --------------------------------------------------------

    recent_90 = (
        segmented
        .assign(
            recent_90=(
                segmented["recency_days"] <= 90
            )
        )
        .groupby("cluster")["recent_90"]
        .mean()
        * 100
    )

    inactive_365 = (
        segmented
        .assign(
            inactive_365=(
                segmented["recency_days"] > 365
            )
        )
        .groupby("cluster")["inactive_365"]
        .mean()
        * 100
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print(model_name)
    print("=" * 75)

    print(
        f"\nSilhouette score: "
        f"{silhouette:.4f}"
    )

    print(
        f"Inertia: "
        f"{inertia:,.2f}"
    )

    print("\nCluster sizes:")

    for cluster in cluster_sizes.index:

        print(
            f"Cluster {cluster}: "
            f"{cluster_sizes[cluster]:,} "
            f"({cluster_pct[cluster]:.2f}%)"
        )

    print("\nAverage profiles:")

    print(
        profiles.to_string()
    )

    print("\nRevenue contribution:")

    for cluster in revenue.index:

        print(
            f"Cluster {cluster}: "
            f"{revenue_pct[cluster]:.2f}%"
        )

    print("\nRecency behavior:")

    for cluster in recent_90.index:

        print(
            f"Cluster {cluster}: "
            f"{recent_90[cluster]:.2f}% within 90 days | "
            f"{inactive_365[cluster]:.2f}% inactive 365+ days"
        )

    return {
        "model": model_name,
        "silhouette": silhouette,
        "inertia": inertia,
        "cluster_sizes": cluster_sizes
    }


# ============================================================
# MODEL A
# ============================================================

model_a = evaluate_model(
    features_with_returns,
    "MODEL A — WITH RETURN RATE"
)


# ============================================================
# MODEL B
# ============================================================

model_b = evaluate_model(
    features_without_returns,
    "MODEL B — WITHOUT RETURN RATE"
)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    f"\nModel A silhouette: "
    f"{model_a['silhouette']:.4f}"
)

print(
    f"Model B silhouette: "
    f"{model_b['silhouette']:.4f}"
)

difference = (
    model_b["silhouette"] -
    model_a["silhouette"]
)

print(
    f"\nSilhouette difference "
    f"(B - A): {difference:+.4f}"
)

print("\nComparison complete.")