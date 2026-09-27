import pandas as pd


# -----------------------------
# Load datasets
# -----------------------------

customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
transactions = pd.read_csv("data/raw/transactions_v3.csv")


# -----------------------------
# Basic dataset information
# -----------------------------

print("=" * 60)
print("DATASET SHAPES")
print("=" * 60)

print(f"Customers:    {customers.shape}")
print(f"Products:     {products.shape}")
print(f"Transactions: {transactions.shape}")


# -----------------------------
# Missing values
# -----------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print("\nCustomers:")
print(customers.isnull().sum())

print("\nProducts:")
print(products.isnull().sum())

print("\nTransactions:")
print(transactions.isnull().sum())


# -----------------------------
# Duplicate IDs
# -----------------------------

print("\n" + "=" * 60)
print("DUPLICATE IDs")
print("=" * 60)

print(
    f"Duplicate customer IDs: "
    f"{customers['customer_id'].duplicated().sum()}"
)

print(
    f"Duplicate product IDs: "
    f"{products['product_id'].duplicated().sum()}"
)

print(
    f"Duplicate transaction IDs: "
    f"{transactions['transaction_id'].duplicated().sum()}"
)


# -----------------------------
# Check customer references
# -----------------------------

print("\n" + "=" * 60)
print("CUSTOMER REFERENCE CHECK")
print("=" * 60)

invalid_customers = ~transactions["customer_id"].isin(
    customers["customer_id"]
)

print(
    f"Transactions with invalid customer IDs: "
    f"{invalid_customers.sum()}"
)


# -----------------------------
# Check product references
# -----------------------------

print("\n" + "=" * 60)
print("PRODUCT REFERENCE CHECK")
print("=" * 60)

invalid_products = ~transactions["product_id"].isin(
    products["product_id"]
)

print(
    f"Transactions with invalid product IDs: "
    f"{invalid_products.sum()}"
)


# -----------------------------
# Check prices
# -----------------------------

print("\n" + "=" * 60)
print("PRICE CHECK")
print("=" * 60)

print(
    f"Products with invalid prices: "
    f"{(products['price'] <= 0).sum()}"
)

print(
    f"Transactions with invalid unit prices: "
    f"{(transactions['unit_price'] <= 0).sum()}"
)


# -----------------------------
# Check quantities
# -----------------------------

print("\n" + "=" * 60)
print("QUANTITY CHECK")
print("=" * 60)

print(
    f"Transactions with invalid quantities: "
    f"{(transactions['quantity'] <= 0).sum()}"
)


# -----------------------------
# Check order statuses
# -----------------------------

print("\n" + "=" * 60)
print("ORDER STATUS CHECK")
print("=" * 60)

print(transactions["order_status"].value_counts())


# -----------------------------
# Check transaction dates
# -----------------------------

print("\n" + "=" * 60)
print("TRANSACTION DATE CHECK")
print("=" * 60)

transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"]
)

print(
    f"Earliest transaction: "
    f"{transactions['transaction_date'].min()}"
)

print(
    f"Latest transaction: "
    f"{transactions['transaction_date'].max()}"
)


print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)