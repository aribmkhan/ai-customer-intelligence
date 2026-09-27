import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of customers
N_CUSTOMERS = 20_000

# Generate customer IDs
customer_ids = [f"CUST{i:05d}" for i in range(1, N_CUSTOMERS + 1)]

# Generate signup dates
signup_dates = pd.to_datetime(
    np.random.choice(
        pd.date_range("2022-01-01", "2025-12-31"),
        size=N_CUSTOMERS
    )
)

# Generate customer demographics
ages = np.random.randint(18, 76, size=N_CUSTOMERS)

genders = np.random.choice(
    ["Male", "Female", "Non-Binary"],
    size=N_CUSTOMERS,
    p=[0.48, 0.48, 0.04]
)

cities = np.random.choice(
    [
        "Toronto",
        "Ottawa",
        "Montreal",
        "Vancouver",
        "Calgary",
        "Edmonton",
        "Winnipeg",
        "Halifax",
        "Quebec City",
        "Hamilton"
    ],
    size=N_CUSTOMERS
)

provinces = np.random.choice(
    [
        "Ontario",
        "Quebec",
        "British Columbia",
        "Alberta",
        "Manitoba",
        "Nova Scotia"
    ],
    size=N_CUSTOMERS
)

acquisition_channels = np.random.choice(
    [
        "Organic Search",
        "Paid Search",
        "Social Media",
        "Email",
        "Referral",
        "Direct"
    ],
    size=N_CUSTOMERS,
    p=[0.25, 0.20, 0.20, 0.12, 0.10, 0.13]
)

device_types = np.random.choice(
    ["Mobile", "Desktop", "Tablet"],
    size=N_CUSTOMERS,
    p=[0.60, 0.32, 0.08]
)

# Create customer dataframe
customers = pd.DataFrame({
    "customer_id": customer_ids,
    "signup_date": signup_dates,
    "age": ages,
    "gender": genders,
    "city": cities,
    "province": provinces,
    "acquisition_channel": acquisition_channels,
    "device_type": device_types
})

# Sort by customer ID
customers = customers.sort_values("customer_id")

# Save to raw data folder
customers.to_csv(
    "data/raw/customers.csv",
    index=False
)

print("Customer dataset created successfully!")
print(f"Number of customers: {len(customers):,}")
print("\nFirst 5 customers:")
print(customers.head())