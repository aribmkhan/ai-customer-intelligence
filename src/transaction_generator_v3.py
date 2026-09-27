import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
N_TRANSACTIONS = 125_000

START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2026-08-30")

CUSTOMERS_PATH = Path("data/raw/customers.csv")
PRODUCTS_PATH = Path("data/raw/products.csv")
OUTPUT_PATH = Path("data/raw/transactions_v3.csv")

np.random.seed(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

customers = pd.read_csv(CUSTOMERS_PATH)
products = pd.read_csv(PRODUCTS_PATH)

customers["signup_date"] = pd.to_datetime(customers["signup_date"])

print(f"Loaded {len(customers):,} customers")
print(f"Loaded {len(products):,} products")


# ============================================================
# CREATE CUSTOMER BEHAVIOR PROFILES
# ============================================================
# We intentionally create different behavioral tendencies.
# The ML model will later attempt to DISCOVER these patterns
# from transaction behavior.

behavior_types = np.random.choice(
    [
        "high_value",
        "loyal",
        "frequent_low_value",
        "at_risk",
        "occasional",
        "new_customer"
    ],
    size=len(customers),
    p=[
        0.10,
        0.20,
        0.15,
        0.15,
        0.30,
        0.10
    ]
)

customers["behavior_type"] = behavior_types


# ============================================================
# BEHAVIOR PARAMETERS
# ============================================================

behavior_params = {
    "high_value": {
        "activity_weight": 2.5,
        "spend_multiplier": 2.5,
        "recency_bias": 0.85,
    },
    "loyal": {
        "activity_weight": 2.0,
        "spend_multiplier": 1.3,
        "recency_bias": 0.90,
    },
    "frequent_low_value": {
        "activity_weight": 2.8,
        "spend_multiplier": 0.55,
        "recency_bias": 0.92,
    },
    "at_risk": {
        "activity_weight": 1.2,
        "spend_multiplier": 1.5,
        "recency_bias": 0.35,
    },
    "occasional": {
        "activity_weight": 0.8,
        "spend_multiplier": 1.0,
        "recency_bias": 0.65,
    },
    "new_customer": {
        "activity_weight": 0.5,
        "spend_multiplier": 1.0,
        "recency_bias": 0.95,
    },
}


# ============================================================
# DETERMINE NUMBER OF TRANSACTIONS PER CUSTOMER
# ============================================================

weights = np.array([
    behavior_params[b]["activity_weight"]
    for b in customers["behavior_type"]
])

# Base transaction allocation
raw_counts = np.random.gamma(
    shape=2.0,
    scale=3.0,
    size=len(customers)
)

weighted_counts = raw_counts * weights

# Normalize to exactly N_TRANSACTIONS
transaction_counts = (
    weighted_counts / weighted_counts.sum() * N_TRANSACTIONS
).astype(int)

# Guarantee at least some activity for most customers
transaction_counts = np.maximum(transaction_counts, 1)

# Adjust total
difference = N_TRANSACTIONS - transaction_counts.sum()

if difference > 0:
    indices = np.random.choice(
        len(customers),
        size=difference,
        replace=True
    )
    np.add.at(transaction_counts, indices, 1)

elif difference < 0:
    removable = np.where(transaction_counts > 1)[0]

    indices = np.random.choice(
        removable,
        size=abs(difference),
        replace=False
    )

    transaction_counts[indices] -= 1


customers["transaction_count"] = transaction_counts


# ============================================================
# GENERATE TRANSACTIONS
# ============================================================

transactions = []

transaction_id = 1

for _, customer in customers.iterrows():

    customer_id = customer["customer_id"]
    signup_date = customer["signup_date"]
    behavior = customer["behavior_type"]

    count = int(customer["transaction_count"])

    params = behavior_params[behavior]

    # --------------------------------------------------------
    # CUSTOMER-SPECIFIC ACTIVITY WINDOW
    # --------------------------------------------------------

    # New customers should generally purchase later.
    if behavior == "new_customer":

        earliest_date = max(
            signup_date,
            END_DATE - pd.Timedelta(days=450)
        )

    else:

        earliest_date = max(signup_date, START_DATE)

    if earliest_date >= END_DATE:
        earliest_date = END_DATE - pd.Timedelta(days=30)

    available_days = max(
        1,
        (END_DATE - earliest_date).days
    )

    # --------------------------------------------------------
    # PURCHASE TIMING
    # --------------------------------------------------------
    # Different behaviors receive different temporal patterns.

    if behavior == "high_value":

        # Recent and relatively consistent purchasing
        beta_a = 2.5
        beta_b = 1.8

    elif behavior == "loyal":

        # Consistent activity throughout their lifetime
        beta_a = 2.0
        beta_b = 2.0

    elif behavior == "frequent_low_value":

        # Frequent purchases, including recent activity
        beta_a = 2.2
        beta_b = 1.8

    elif behavior == "at_risk":

        # Concentrate activity earlier in customer lifetime
        beta_a = 1.2
        beta_b = 3.5

    elif behavior == "occasional":

        beta_a = 1.3
        beta_b = 2.5

    else:  # new_customer

        beta_a = 2.5
        beta_b = 1.5

    time_position = np.random.beta(
        beta_a,
        beta_b,
        size=count
    )

    purchase_days = (
        time_position * available_days
    ).astype(int)

    purchase_dates = [
        earliest_date + pd.Timedelta(days=int(day))
        for day in purchase_days
    ]

    # --------------------------------------------------------
    # ENSURE AT LEAST ONE RECENT PURCHASE FOR ACTIVE USERS
    # --------------------------------------------------------

    if behavior in ["high_value", "loyal", "frequent_low_value"]:

        recent_index = np.random.randint(0, count)

        recent_days = np.random.randint(
            0,
            min(180, available_days) + 1
        )

        purchase_dates[recent_index] = (
            END_DATE - pd.Timedelta(days=int(recent_days))
        )

        # Don't allow purchase before signup
        if purchase_dates[recent_index] < earliest_date:
            purchase_dates[recent_index] = earliest_date

    # --------------------------------------------------------
    # PRODUCT SELECTION
    # --------------------------------------------------------

    product_ids = np.random.choice(
        products["product_id"],
        size=count,
        replace=True
    )

    # --------------------------------------------------------
    # QUANTITY
    # --------------------------------------------------------

    quantities = np.random.choice(
        [1, 2, 3, 4],
        size=count,
        p=[0.70, 0.20, 0.08, 0.02]
    )

    # --------------------------------------------------------
    # PRODUCT PRICES
    # --------------------------------------------------------

    product_lookup = products.set_index("product_id")

    base_prices = np.array([
        product_lookup.loc[pid, "price"]
        for pid in product_ids
    ])

    # Customer-specific spending behavior
    spend_noise = np.random.lognormal(
        mean=0,
        sigma=0.20,
        size=count
    )

    unit_prices = (
        base_prices
        * params["spend_multiplier"]
        * spend_noise
    )

    # Keep realistic price range
    unit_prices = np.clip(
        unit_prices,
        10,
        2500
    )

    # --------------------------------------------------------
    # DISCOUNTS
    # --------------------------------------------------------

    discount_pct = np.random.choice(
        [0, 5, 10, 15, 20, 25],
        size=count,
        p=[0.25, 0.20, 0.25, 0.15, 0.10, 0.05]
    )

    # Frequent low-value customers use discounts more often
    if behavior == "frequent_low_value":

        extra_discount = np.random.choice(
            [0, 5, 10],
            size=count,
            p=[0.40, 0.40, 0.20]
        )

        discount_pct = np.minimum(
            discount_pct + extra_discount,
            30
        )

    # --------------------------------------------------------
    # PAYMENT METHOD
    # --------------------------------------------------------

    payment_methods = np.random.choice(
        [
            "Credit Card",
            "Debit Card",
            "PayPal",
            "Apple Pay",
            "Google Pay"
        ],
        size=count,
        p=[
            0.38,
            0.25,
            0.18,
            0.11,
            0.08
        ]
    )

    # --------------------------------------------------------
    # ORDER STATUS
    # --------------------------------------------------------

    statuses = np.random.choice(
        [
            "Completed",
            "Returned",
            "Cancelled"
        ],
        size=count,
        p=[
            0.92,
            0.05,
            0.03
        ]
    )

    # --------------------------------------------------------
    # CREATE RECORDS
    # --------------------------------------------------------

    for i in range(count):

        transactions.append({
            "transaction_id": f"TXN{transaction_id:07d}",
            "customer_id": customer_id,
            "product_id": product_ids[i],
            "transaction_date": purchase_dates[i],
            "quantity": quantities[i],
            "unit_price": round(unit_prices[i], 2),
            "discount_pct": discount_pct[i],
            "payment_method": payment_methods[i],
            "order_status": statuses[i]
        })

        transaction_id += 1


# ============================================================
# CREATE DATAFRAME
# ============================================================

transactions_df = pd.DataFrame(transactions)

transactions_df["transaction_date"] = pd.to_datetime(
    transactions_df["transaction_date"]
)

transactions_df = transactions_df.sort_values(
    "transaction_date"
).reset_index(drop=True)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("TRANSACTION DATA VALIDATION")
print("=" * 60)

print(f"\nTotal transactions: {len(transactions_df):,}")

print(
    f"Date range: "
    f"{transactions_df['transaction_date'].min().date()} "
    f"to "
    f"{transactions_df['transaction_date'].max().date()}"
)

print("\nTransactions by year:")
print(
    transactions_df["transaction_date"]
    .dt.year
    .value_counts()
    .sort_index()
)

print("\nOrder status:")
print(
    transactions_df["order_status"]
    .value_counts()
)

print("\nTransactions per customer:")
print(
    customers["transaction_count"].describe()
)

print("\nBehavior types:")
print(
    customers["behavior_type"]
    .value_counts()
)


# ============================================================
# SAVE
# ============================================================

transactions_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)