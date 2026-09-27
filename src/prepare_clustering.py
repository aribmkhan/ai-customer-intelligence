import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/processed/customer_metrics.csv")

OUTPUT_PATH = Path(
    "data/processed/clustering_features.csv"
)


# ============================================================
# LOAD CUSTOMER METRICS
# ============================================================

customer_metrics = pd.read_csv(
    INPUT_PATH
)

print(
    f"Loaded {len(customer_metrics):,} customers"
)


# ============================================================
# SELECT CLUSTERING FEATURES
# ============================================================
# We want customer segments to be based primarily on
# purchasing behavior rather than demographics.
#
# total_orders:
#   How frequently the customer purchases
#
# total_revenue:
#   Historical monetary value of the customer
#
# average_order_value:
#   Typical size of a customer's transaction
#
# recency_days:
#   How recently the customer purchased
#
# category_count:
#   Breadth of product categories purchased
#
# average_discount:
#   Customer's typical discount usage
#
# return_rate:
#   Share of transactions returned

feature_columns = [
    "total_orders",
    "total_revenue",
    "average_order_value",
    "recency_days",
    "category_count",
    "average_discount",
    "return_rate"
]

features = customer_metrics[
    ["customer_id"] + feature_columns
].copy()


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(
    features.isnull().sum()
)


# ============================================================
# RAW FEATURE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 60)
print("RAW FEATURE SUMMARY")
print("=" * 60)

print(
    features[
        feature_columns
    ]
    .describe()
    .round(2)
)


# ============================================================
# SKEWNESS BEFORE TRANSFORMATION
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SKEWNESS BEFORE TRANSFORMATION")
print("=" * 60)

print(
    features[
        feature_columns
    ]
    .skew()
    .sort_values(ascending=False)
    .round(2)
)


# ============================================================
# LOG TRANSFORM HIGHLY SKEWED FEATURES
# ============================================================
# log1p(x) = log(1 + x)
#
# It compresses large values while preserving ordering.
# This reduces the influence of extreme high-value customers.
#
# We transform variables where large right tails are expected.

log_features = [
    "total_orders",
    "total_revenue",
    "average_order_value",
    "recency_days"
]

for column in log_features:

    features[f"log_{column}"] = np.log1p(
        features[column]
    )


# ============================================================
# FEATURES USED BY K-MEANS
# ============================================================

model_features = [
    "log_total_orders",
    "log_total_revenue",
    "log_average_order_value",
    "log_recency_days",
    "category_count",
    "average_discount",
    "return_rate"
]


# ============================================================
# CHECK SKEWNESS AFTER TRANSFORMATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL FEATURE SKEWNESS")
print("=" * 60)

print(
    features[
        model_features
    ]
    .skew()
    .sort_values(ascending=False)
    .round(2)
)


# ============================================================
# STANDARDIZE FEATURES
# ============================================================
# K-Means uses Euclidean distance.
#
# Without scaling, variables with larger numerical ranges
# would have more influence over cluster assignment.
#
# StandardScaler transforms each variable approximately to:
#
# mean = 0
# standard deviation = 1

scaler = StandardScaler()

scaled_values = scaler.fit_transform(
    features[model_features]
)


# ============================================================
# CREATE SCALED DATAFRAME
# ============================================================

scaled_column_names = [
    f"scaled_{column}"
    for column in model_features
]

scaled_df = pd.DataFrame(
    scaled_values,
    columns=scaled_column_names
)

scaled_df.insert(
    0,
    "customer_id",
    features["customer_id"].values
)


# ============================================================
# VALIDATE SCALED FEATURES
# ============================================================

print("\n" + "=" * 60)
print("SCALED FEATURE SUMMARY")
print("=" * 60)

print(
    scaled_df[
        scaled_column_names
    ]
    .describe()
    .round(2)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

scaled_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved clustering features to: "
    f"{OUTPUT_PATH}"
)

print(
    f"Shape: {scaled_df.shape}"
)