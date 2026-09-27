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
    "data/processed/final_customer_segments.csv"
)

PROFILE_PATH = Path(
    "data/processed/final_segment_profiles.csv"
)

N_CLUSTERS = 3
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

features = pd.read_csv(
    FEATURES_PATH
)

customer_metrics = pd.read_csv(
    METRICS_PATH
)

print(
    f"Loaded {len(features):,} clustering records"
)

print(
    f"Loaded {len(customer_metrics):,} customer metric records"
)


# ============================================================
# FINAL MODEL FEATURES
# ============================================================
# return_rate is intentionally excluded from cluster assignment.
# It remains available as a descriptive business KPI.

feature_columns = [
    "scaled_log_total_orders",
    "scaled_log_total_revenue",
    "scaled_log_average_order_value",
    "scaled_log_recency_days",
    "scaled_category_count",
    "scaled_average_discount"
]

X = features[feature_columns]

print(
    f"\nUsing {len(feature_columns)} final model features"
)


# ============================================================
# TRAIN FINAL K-MEANS MODEL
# ============================================================

model = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=20
)

features["cluster"] = model.fit_predict(X)


# ============================================================
# JOIN CLUSTERS TO ORIGINAL CUSTOMER METRICS
# ============================================================

segmented = customer_metrics.merge(
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
# INSPECT RAW CLUSTER PROFILES
# ============================================================

raw_profiles = (
    segmented
    .groupby("cluster")
    .agg(
        customer_count=(
            "customer_id",
            "count"
        ),
        avg_orders=(
            "total_orders",
            "mean"
        ),
        avg_revenue=(
            "total_revenue",
            "mean"
        ),
        avg_order_value=(
            "average_order_value",
            "mean"
        ),
        avg_recency_days=(
            "recency_days",
            "mean"
        ),
        avg_category_count=(
            "category_count",
            "mean"
        ),
        avg_discount=(
            "average_discount",
            "mean"
        )
    )
)

print("\n" + "=" * 75)
print("RAW CLUSTER PROFILES")
print("=" * 75)

print(
    raw_profiles.round(2).to_string()
)


# ============================================================
# IDENTIFY BUSINESS SEGMENTS
# ============================================================
# K-Means cluster numbers are arbitrary.
#
# Instead of assuming that cluster 0, 1, or 2 always represents
# a particular customer type, we identify each segment from its
# observed business characteristics.
#
# High-Value Active:
#     Cluster with highest average total revenue.
#
# Low-Value Discount-Oriented:
#     Of the remaining clusters, the one with the lowest
#     average total revenue.
#
# Higher-Spend Lapsed:
#     Remaining cluster.


high_value_cluster = (
    raw_profiles["avg_revenue"]
    .idxmax()
)

remaining_clusters = [
    cluster
    for cluster in raw_profiles.index
    if cluster != high_value_cluster
]

low_value_cluster = (
    raw_profiles
    .loc[remaining_clusters, "avg_revenue"]
    .idxmin()
)

lapsed_cluster = [
    cluster
    for cluster in remaining_clusters
    if cluster != low_value_cluster
][0]


# ============================================================
# CREATE SEGMENT MAPPING
# ============================================================

segment_mapping = {
    high_value_cluster:
        "High-Value Active",

    low_value_cluster:
        "Low-Value Discount-Oriented",

    lapsed_cluster:
        "Higher-Spend Lapsed"
}

segmented["segment_name"] = (
    segmented["cluster"]
    .map(segment_mapping)
)


print("\n" + "=" * 75)
print("CLUSTER → BUSINESS SEGMENT MAPPING")
print("=" * 75)

for cluster, segment in segment_mapping.items():

    print(
        f"Cluster {cluster} -> {segment}"
    )


# ============================================================
# FINAL SEGMENT PROFILES
# ============================================================

segment_profiles = (
    segmented
    .groupby("segment_name")
    .agg(
        customer_count=(
            "customer_id",
            "count"
        ),
        avg_orders=(
            "total_orders",
            "mean"
        ),
        avg_revenue=(
            "total_revenue",
            "mean"
        ),
        median_revenue=(
            "total_revenue",
            "median"
        ),
        avg_order_value=(
            "average_order_value",
            "mean"
        ),
        avg_recency_days=(
            "recency_days",
            "mean"
        ),
        avg_category_count=(
            "category_count",
            "mean"
        ),
        avg_discount=(
            "average_discount",
            "mean"
        ),
        avg_return_rate=(
            "return_rate",
            "mean"
        )
    )
)


# ============================================================
# CUSTOMER SHARE
# ============================================================

segment_profiles["customer_share_pct"] = (
    segment_profiles["customer_count"]
    / len(segmented)
    * 100
)


# ============================================================
# REVENUE SHARE
# ============================================================

revenue_by_segment = (
    segmented
    .groupby("segment_name")["total_revenue"]
    .sum()
)

segment_profiles["total_revenue"] = (
    revenue_by_segment
)

segment_profiles["revenue_share_pct"] = (
    segment_profiles["total_revenue"]
    / segmented["total_revenue"].sum()
    * 100
)


# ============================================================
# RECENCY KPIS
# ============================================================

recent_90 = (
    segmented
    .assign(
        recent_90=(
            segmented["recency_days"] <= 90
        )
    )
    .groupby("segment_name")["recent_90"]
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
    .groupby("segment_name")["inactive_365"]
    .mean()
    * 100
)

segment_profiles[
    "purchased_within_90_days_pct"
] = recent_90

segment_profiles[
    "inactive_365_plus_days_pct"
] = inactive_365


# ============================================================
# FORMAT PROFILE OUTPUT
# ============================================================

segment_profiles = (
    segment_profiles
    .reset_index()
    .round(2)
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("VALIDATION")
print("=" * 75)

print(
    f"Segmented customers: "
    f"{len(segmented):,}"
)

print(
    f"Unique customers: "
    f"{segmented['customer_id'].nunique():,}"
)

print(
    f"Missing segment names: "
    f"{segmented['segment_name'].isna().sum():,}"
)

print(
    f"Duplicate customer IDs: "
    f"{segmented['customer_id'].duplicated().sum():,}"
)


# ============================================================
# DISPLAY FINAL SEGMENT PROFILES
# ============================================================

print("\n" + "=" * 75)
print("FINAL BUSINESS SEGMENTS")
print("=" * 75)

print(
    segment_profiles.to_string(
        index=False
    )
)


# ============================================================
# SAVE FINAL DATASETS
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

segmented.to_csv(
    OUTPUT_PATH,
    index=False
)

segment_profiles.to_csv(
    PROFILE_PATH,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)

print(
    f"Customer-level segmentation: "
    f"{OUTPUT_PATH}"
)

print(
    f"Segment profiles: "
    f"{PROFILE_PATH}"
)

print(
    "\nFinal customer segmentation complete."
)