import pandas as pd


# ============================================================
# LOAD DATA
# ============================================================

customers = pd.read_csv(
    "data/raw/customers.csv",
    parse_dates=["signup_date"]
)

products = pd.read_csv(
    "data/raw/products.csv"
)

transactions = pd.read_csv(
    "data/raw/transactions_v3.csv",
    parse_dates=["transaction_date"]
)


# ============================================================
# USE COMPLETED TRANSACTIONS FOR SALES ANALYSIS
# ============================================================

completed = transactions[
    transactions["order_status"] == "Completed"
].copy()

completed["gross_sales"] = (
    completed["quantity"] *
    completed["unit_price"]
)

completed["revenue"] = (
    completed["gross_sales"] *
    (1 - completed["discount_pct"] / 100)
)


# ============================================================
# OVERALL BUSINESS KPIs
# ============================================================

print("\n" + "=" * 60)
print("NOVAMART BUSINESS KPIs")
print("=" * 60)

total_customers = customers["customer_id"].nunique()

purchasing_customers = completed[
    "customer_id"
].nunique()

completed_transactions = len(completed)

total_revenue = completed["revenue"].sum()

average_transaction_value = completed[
    "revenue"
].mean()

average_items = completed[
    "quantity"
].mean()

print(f"Total customers: {total_customers:,}")

print(
    f"Customers with completed purchases: "
    f"{purchasing_customers:,}"
)

print(
    f"Completed transactions: "
    f"{completed_transactions:,}"
)

print(
    f"Total revenue: "
    f"${total_revenue:,.2f}"
)

print(
    f"Average transaction value: "
    f"${average_transaction_value:,.2f}"
)

print(
    f"Average items per transaction: "
    f"{average_items:.2f}"
)


# ============================================================
# CUSTOMER ACQUISITION CHANNELS
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER ACQUISITION CHANNELS")
print("=" * 60)

print(
    customers["acquisition_channel"]
    .value_counts()
)


# ============================================================
# REVENUE BY PRODUCT CATEGORY
# ============================================================

sales_with_products = completed.merge(
    products[
        ["product_id", "product_name", "category", "subcategory"]
    ],
    on="product_id",
    how="left"
)

category_revenue = (
    sales_with_products
    .groupby("category")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("REVENUE BY CATEGORY")
print("=" * 60)

print(
    category_revenue.apply(
        lambda x: f"${x:,.2f}"
    )
)


# ============================================================
# TOP PRODUCTS
# ============================================================

product_revenue = (
    sales_with_products
    .groupby(
        ["product_id", "product_name", "category"]
    )["revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("TOP 10 PRODUCTS BY REVENUE")
print("=" * 60)

print(
    product_revenue
    .head(10)
    .apply(lambda x: f"${x:,.2f}")
)


# ============================================================
# CUSTOMER-LEVEL REVENUE
# ============================================================

customer_revenue = (
    completed
    .groupby("customer_id")["revenue"]
    .sum()
)

print("\n" + "=" * 60)
print("CUSTOMER REVENUE DISTRIBUTION")
print("=" * 60)

print(
    customer_revenue
    .describe()
    .round(2)
)


# ============================================================
# TOP CUSTOMERS
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 CUSTOMERS BY REVENUE")
print("=" * 60)

print(
    customer_revenue
    .sort_values(ascending=False)
    .head(10)
    .apply(lambda x: f"${x:,.2f}")
)


# ============================================================
# PURCHASE FREQUENCY
# ============================================================

customer_orders = (
    completed
    .groupby("customer_id")["transaction_id"]
    .count()
)

print("\n" + "=" * 60)
print("COMPLETED TRANSACTIONS PER CUSTOMER")
print("=" * 60)

print(
    customer_orders
    .describe()
    .round(2)
)


# ============================================================
# MONTHLY REVENUE
# ============================================================

monthly_revenue = (
    completed
    .set_index("transaction_date")
    .resample("ME")["revenue"]
    .sum()
)

print("\n" + "=" * 60)
print("MONTHLY REVENUE - LAST 12 MONTHS")
print("=" * 60)

for date, revenue in monthly_revenue.tail(12).items():

    print(
        f"{date.strftime('%Y-%m')}: "
        f"${revenue:,.2f}"
    )


# ============================================================
# RECENCY
# ============================================================

analysis_date = pd.Timestamp("2026-08-31")

last_purchase = (
    completed
    .groupby("customer_id")["transaction_date"]
    .max()
)

recency_days = (
    analysis_date - last_purchase
).dt.days

print("\n" + "=" * 60)
print("CUSTOMER RECENCY DISTRIBUTION")
print("=" * 60)

print(
    recency_days
    .describe()
    .round(2)
)


# ============================================================
# RETURN BEHAVIOR
# ============================================================

returned = transactions[
    transactions["order_status"] == "Returned"
]

return_rate = (
    len(returned) / len(transactions)
) * 100

print("\n" + "=" * 60)
print("RETURN BEHAVIOR")
print("=" * 60)

print(
    f"Returned transactions: "
    f"{len(returned):,}"
)

print(
    f"Overall return rate: "
    f"{return_rate:.2f}%"
)


# ============================================================
# YEARLY TRANSACTION ACTIVITY
# ============================================================

yearly_activity = (
    transactions
    .groupby(
        transactions["transaction_date"].dt.year
    )
    .size()
)

print("\n" + "=" * 60)
print("TRANSACTIONS BY YEAR")
print("=" * 60)

print(yearly_activity)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)