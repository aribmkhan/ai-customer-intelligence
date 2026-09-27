import pandas as pd

from pathlib import Path
from sklearn.cluster import KMeans


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_PATH = Path(
    "data/processed/clustering_features.csv"
)

METRICS_PATH = Path(
    "data/processed/customer_metrics.csv"
)

OUTPUT_PATH = Path(
    "data/processed/customer_segments.csv"
)

N_CLUSTERS = 3
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

features = pd.read_csv(FEATURES_PATH)

customer_metrics = pd.read_csv(
    METRICS_PATH,
    parse_dates=[
        "signup_date",
        "first_purchase_date",
        "last_purchase_date"
    ]
)

print(
    f"Loaded {len(features):,} clustering records"
)

print(
    f"Loaded {len(customer_metrics):,} customer metrics records"
)


# ============================================================
# SELECT SCALED MODEL FEATURES
# ============================================================

feature_columns = [
    column
    for column in features.columns
    if column.startswith("scaled_")
]

X = features[feature_columns]

print(
    f"\nUsing {len(feature_columns)} features"
)


# ============================================================
# FIT K-MEANS MODEL
# ============================================================

model = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=20
)

cluster_labels = model.fit_predict(X)

features["cluster"] = cluster_labels


# ============================================================
# JOIN CLUSTERS TO CUSTOMER METRICS
# ============================================================

segmented_customers = customer_metrics.merge(
    features[
        [
            "customer_id",
            "cluster"
        ]
    ],
    on="customer_id",
    how="inner"
)


# ============================================================
# CLUSTER SIZES
# ============================================================

cluster_sizes = (
    segmented_customers["cluster"]
    .value_counts()
    .sort_index()
)

cluster_percentages = (
    segmented_customers["cluster"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\n" + "=" * 70)
print("CLUSTER SIZES")
print("=" * 70)

for cluster in cluster_sizes.index:

    print(
        f"Cluster {cluster}: "
        f"{cluster_sizes[cluster]:,} customers "
        f"({cluster_percentages[cluster]:.2f}%)"
    )


# ============================================================
# CLUSTER BEHAVIOR PROFILES
# ============================================================

profile_columns = [
    "total_orders",
    "total_revenue",
    "average_order_value",
    "recency_days",
    "total_items",
    "average_discount",
    "category_count",
    "return_rate",
    "customer_tenure_days",
    "annualized_customer_value"
]

cluster_profiles = (
    segmented_customers
    .groupby("cluster")[profile_columns]
    .mean()
    .round(2)
)

print("\n" + "=" * 70)
print("AVERAGE CLUSTER PROFILES")
print("=" * 70)

print(
    cluster_profiles.to_string()
)


# ============================================================
# CLUSTER MEDIANS
# ============================================================
# Means can be influenced by high-value outliers.
# Medians give us another view of the typical customer.

cluster_medians = (
    segmented_customers
    .groupby("cluster")[profile_columns]
    .median()
    .round(2)
)

print("\n" + "=" * 70)
print("MEDIAN CLUSTER PROFILES")
print("=" * 70)

print(
    cluster_medians.to_string()
)


# ============================================================
# REVENUE CONTRIBUTION
# ============================================================

revenue_by_cluster = (
    segmented_customers
    .groupby("cluster")["total_revenue"]
    .sum()
)

total_revenue = (
    segmented_customers["total_revenue"]
    .sum()
)

revenue_share = (
    revenue_by_cluster /
    total_revenue *
    100
)

print("\n" + "=" * 70)
print("REVENUE CONTRIBUTION BY CLUSTER")
print("=" * 70)

for cluster in revenue_by_cluster.index:

    print(
        f"Cluster {cluster}: "
        f"${revenue_by_cluster[cluster]:,.2f} "
        f"({revenue_share[cluster]:.2f}% of revenue)"
    )


# ============================================================
# RECENT CUSTOMER ACTIVITY
# ============================================================

print("\n" + "=" * 70)
print("RECENCY BREAKDOWN")
print("=" * 70)

for cluster in sorted(
    segmented_customers["cluster"].unique()
):

    cluster_data = segmented_customers[
        segmented_customers["cluster"] == cluster
    ]

    recent_90 = (
        cluster_data["recency_days"] <= 90
    ).mean() * 100

    inactive_365 = (
        cluster_data["recency_days"] > 365
    ).mean() * 100

    print(
        f"\nCluster {cluster}:"
    )

    print(
        f"  Purchased within 90 days: "
        f"{recent_90:.2f}%"
    )

    print(
        f"  No purchase in 365+ days: "
        f"{inactive_365:.2f}%"
    )


# ============================================================
# ACQUISITION CHANNEL MIX
# ============================================================

channel_mix = pd.crosstab(
    segmented_customers["cluster"],
    segmented_customers["acquisition_channel"],
    normalize="index"
) * 100

print("\n" + "=" * 70)
print("ACQUISITION CHANNEL MIX (%)")
print("=" * 70)

print(
    channel_mix.round(2).to_string()
)


# ============================================================
# DEVICE MIX
# ============================================================

device_mix = pd.crosstab(
    segmented_customers["cluster"],
    segmented_customers["device_type"],
    normalize="index"
) * 100

print("\n" + "=" * 70)
print("DEVICE MIX (%)")
print("=" * 70)

print(
    device_mix.round(2).to_string()
)


# ============================================================
# SAVE CLUSTER PROFILE SUMMARY
# ============================================================

profile_output = cluster_profiles.copy()

profile_output["customer_count"] = cluster_sizes

profile_output["customer_share_pct"] = (
    cluster_percentages
)

profile_output["revenue_share_pct"] = (
    revenue_share
)

profile_output = profile_output.reset_index()

profile_path = Path(
    "data/processed/cluster_profiles.csv"
)

profile_output.to_csv(
    profile_path,
    index=False
)


# ============================================================
# SAVE CUSTOMER SEGMENTS
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

segmented_customers.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"Customer segments: {OUTPUT_PATH}"
)

print(
    f"Cluster profiles: {profile_path}"
)

print("\nK-Means segmentation complete.")