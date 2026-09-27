import pandas as pd
import numpy as np


# -----------------------------
# Load datasets
# -----------------------------

customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
transactions = pd.read_csv("data/raw/transactions.csv")


# -----------------------------
# Prepare data
# -----------------------------

transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"]
)

transactions["revenue"] = (
    transactions["quantity"]
    * transactions["unit_price"]
    * (1 - transactions["discount_pct"])
)


# -----------------------------
# Overall business metrics
# -----------------------------

completed = transactions[
    transactions["order_status"] == "Completed"
].copy()

print("=" * 60)
print("NOVAMART BUSINESS OVERVIEW")
print("=" * 60)

print(f"Total customers: {len(customers):,}")

print(
    f"Customers with purchases: "
    f"{completed['customer_id'].nunique():,}"
)

print(f"Completed orders: {len(completed):,}")

print(
    f"Total revenue: "
    f"${completed['revenue'].sum():,.2f}"
)

print(
    f"Average transaction value: "
    f"${completed['revenue'].mean():,.2f}"
)

print(
    f"Average items per transaction: "
    f"{completed['quantity'].mean():.2f}"
)


# -----------------------------
# Customer acquisition
# -----------------------------

print("\n" + "=" * 60)
print("ACQUISITION CHANNELS")
print("=" * 60)

print(
    customers["acquisition_channel"]
    .value_counts()
)


# -----------------------------
# Product categories
# -----------------------------

print("\n" + "=" * 60)
print("REVENUE BY PRODUCT CATEGORY")
print("=" * 60)

category_revenue = (
    completed
    .merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    )
    .groupby("category")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

print(category_revenue)


# -----------------------------
# Top products
# -----------------------------

print("\n" + "=" * 60)
print("TOP 10 PRODUCTS BY REVENUE")
print("=" * 60)

product_revenue = (
    completed
    .groupby("product_id")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_products = (
    product_revenue
    .reset_index()
    .merge(
        products[["product_id", "product_name"]],
        on="product_id"
    )
)

print(top_products)


# -----------------------------
# Customer revenue
# -----------------------------

print("\n" + "=" * 60)
print("CUSTOMER REVENUE DISTRIBUTION")
print("=" * 60)

customer_revenue = (
    completed
    .groupby("customer_id")["revenue"]
    .sum()
)

print(
    customer_revenue.describe()
)


# -----------------------------
# Top customers
# -----------------------------

print("\n" + "=" * 60)
print("TOP 10 CUSTOMERS BY REVENUE")
print("=" * 60)

print(
    customer_revenue
    .sort_values(ascending=False)
    .head(10)
)


# -----------------------------
# Purchase frequency
# -----------------------------

print("\n" + "=" * 60)
print("ORDERS PER CUSTOMER")
print("=" * 60)

orders_per_customer = (
    completed
    .groupby("customer_id")
    .size()
)

print(
    orders_per_customer.describe()
)


# -----------------------------
# Monthly revenue
# -----------------------------

print("\n" + "=" * 60)
print("MONTHLY REVENUE")
print("=" * 60)

monthly_revenue = (
    completed
    .set_index("transaction_date")
    .resample("ME")["revenue"]
    .sum()
)

print(monthly_revenue.tail(12))


print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)