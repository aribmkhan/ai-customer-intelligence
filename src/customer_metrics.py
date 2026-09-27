import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

CUSTOMERS_PATH = Path("data/raw/customers.csv")
PRODUCTS_PATH = Path("data/raw/products.csv")
TRANSACTIONS_PATH = Path("data/raw/transactions_v3.csv")

OUTPUT_PATH = Path("data/processed/customer_metrics.csv")

ANALYSIS_DATE = pd.Timestamp("2026-08-31")


# ============================================================
# LOAD DATA
# ============================================================

customers = pd.read_csv(
    CUSTOMERS_PATH,
    parse_dates=["signup_date"]
)

products = pd.read_csv(
    PRODUCTS_PATH
)

transactions = pd.read_csv(
    TRANSACTIONS_PATH,
    parse_dates=["transaction_date"]
)

print(f"Loaded {len(customers):,} customers")
print(f"Loaded {len(products):,} products")
print(f"Loaded {len(transactions):,} transactions")


# ============================================================
# CALCULATE TRANSACTION REVENUE
# ============================================================

transactions["gross_sales"] = (
    transactions["quantity"] *
    transactions["unit_price"]
)

transactions["revenue"] = (
    transactions["gross_sales"] *
    (1 - transactions["discount_pct"] / 100)
)


# ============================================================
# COMPLETED TRANSACTIONS
# ============================================================

completed = transactions[
    transactions["order_status"] == "Completed"
].copy()


# ============================================================
# ADD PRODUCT INFORMATION
# ============================================================

completed = completed.merge(
    products[
        [
            "product_id",
            "category",
            "subcategory"
        ]
    ],
    on="product_id",
    how="left"
)


# ============================================================
# CORE CUSTOMER METRICS
# ============================================================

customer_metrics = (
    completed
    .groupby("customer_id")
    .agg(
        total_orders=(
            "transaction_id",
            "count"
        ),
        total_revenue=(
            "revenue",
            "sum"
        ),
        average_order_value=(
            "revenue",
            "mean"
        ),
        first_purchase_date=(
            "transaction_date",
            "min"
        ),
        last_purchase_date=(
            "transaction_date",
            "max"
        ),
        total_items=(
            "quantity",
            "sum"
        ),
        average_discount=(
            "discount_pct",
            "mean"
        ),
        category_count=(
            "category",
            "nunique"
        )
    )
    .reset_index()
)


# ============================================================
# RECENCY
# ============================================================

customer_metrics["recency_days"] = (
    ANALYSIS_DATE -
    customer_metrics["last_purchase_date"]
).dt.days


# ============================================================
# PURCHASE FREQUENCY
# ============================================================
# Frequency is defined here as completed transactions.

customer_metrics["purchase_frequency"] = (
    customer_metrics["total_orders"]
)


# ============================================================
# CUSTOMER TENURE
# ============================================================

customer_metrics["customer_tenure_days"] = (
    ANALYSIS_DATE -
    customer_metrics["first_purchase_date"]
).dt.days

customer_metrics["customer_tenure_days"] = (
    customer_metrics["customer_tenure_days"]
    .clip(lower=1)
)


# ============================================================
# RETURN RATE
# ============================================================

all_orders = (
    transactions
    .groupby("customer_id")
    .size()
    .rename("all_orders")
)

returned_orders = (
    transactions[
        transactions["order_status"] == "Returned"
    ]
    .groupby("customer_id")
    .size()
    .rename("returned_orders")
)

return_metrics = pd.concat(
    [
        all_orders,
        returned_orders
    ],
    axis=1
).fillna(0)

return_metrics["return_rate"] = (
    return_metrics["returned_orders"] /
    return_metrics["all_orders"]
)

return_metrics = (
    return_metrics[
        ["return_rate"]
    ]
    .reset_index()
)

customer_metrics = customer_metrics.merge(
    return_metrics,
    on="customer_id",
    how="left"
)


# ============================================================
# SIMPLE HISTORICAL CUSTOMER VALUE
# ============================================================
# This is not a predictive CLV model.
# It normalizes historical revenue by customer tenure
# and annualizes it to provide a comparable value metric.

customer_metrics["annualized_customer_value"] = (
    customer_metrics["total_revenue"] /
    customer_metrics["customer_tenure_days"] *
    365
)


# ============================================================
# ADD CUSTOMER ATTRIBUTES
# ============================================================

customer_metrics = customer_metrics.merge(
    customers[
        [
            "customer_id",
            "signup_date",
            "age",
            "gender",
            "city",
            "province",
            "acquisition_channel",
            "device_type"
        ]
    ],
    on="customer_id",
    how="left"
)


# ============================================================
# ROUND NUMERIC VALUES
# ============================================================

customer_metrics[
    "total_revenue"
] = customer_metrics[
    "total_revenue"
].round(2)

customer_metrics[
    "average_order_value"
] = customer_metrics[
    "average_order_value"
].round(2)

customer_metrics[
    "average_discount"
] = customer_metrics[
    "average_discount"
].round(2)

customer_metrics[
    "return_rate"
] = customer_metrics[
    "return_rate"
].round(4)

customer_metrics[
    "annualized_customer_value"
] = customer_metrics[
    "annualized_customer_value"
].round(2)


# ============================================================
# REORDER COLUMNS
# ============================================================

column_order = [
    "customer_id",
    "signup_date",
    "age",
    "gender",
    "city",
    "province",
    "acquisition_channel",
    "device_type",
    "total_orders",
    "total_revenue",
    "average_order_value",
    "first_purchase_date",
    "last_purchase_date",
    "recency_days",
    "purchase_frequency",
    "total_items",
    "average_discount",
    "category_count",
    "return_rate",
    "customer_tenure_days",
    "annualized_customer_value"
]

customer_metrics = customer_metrics[
    column_order
]


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER METRICS VALIDATION")
print("=" * 60)

print(
    f"\nCustomers in metrics table: "
    f"{len(customer_metrics):,}"
)

print(
    f"Unique customer IDs: "
    f"{customer_metrics['customer_id'].nunique():,}"
)

print(
    f"Duplicate customer IDs: "
    f"{customer_metrics['customer_id'].duplicated().sum()}"
)

print(
    f"Missing values: "
    f"{customer_metrics.isnull().sum().sum()}"
)


# ============================================================
# METRIC SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER METRIC SUMMARY")
print("=" * 60)

summary_columns = [
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

print(
    customer_metrics[
        summary_columns
    ]
    .describe()
    .round(2)
)


# ============================================================
# PREVIEW
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER METRICS PREVIEW")
print("=" * 60)

print(
    customer_metrics
    .head(10)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

customer_metrics.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved customer metrics to: "
    f"{OUTPUT_PATH}"
)