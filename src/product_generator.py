import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of products
N_PRODUCTS = 50

# Product categories
categories = {
    "Electronics": [
        "Headphones",
        "Smart Watches",
        "Speakers",
        "Chargers",
        "Computer Accessories"
    ],
    "Home": [
        "Kitchen",
        "Furniture",
        "Lighting",
        "Storage",
        "Home Decor"
    ],
    "Clothing": [
        "Shirts",
        "Pants",
        "Jackets",
        "Shoes",
        "Accessories"
    ],
    "Beauty": [
        "Skincare",
        "Haircare",
        "Makeup",
        "Fragrance",
        "Body Care"
    ],
    "Sports": [
        "Fitness",
        "Running",
        "Camping",
        "Cycling",
        "Outdoor"
    ]
}

# Generate product IDs
product_ids = [
    f"PROD{i:03d}"
    for i in range(1, N_PRODUCTS + 1)
]

# Generate categories
category_list = list(categories.keys())

product_categories = np.random.choice(
    category_list,
    size=N_PRODUCTS
)

# Generate subcategories based on category
subcategories = []

for category in product_categories:
    subcategories.append(
        np.random.choice(categories[category])
    )

# Generate product names
product_names = [
    f"{subcategory} Product {i:02d}"
    for i, subcategory in enumerate(subcategories, start=1)
]

# Generate realistic prices
prices = np.round(
    np.random.uniform(15, 500, size=N_PRODUCTS),
    2
)

# Create dataframe
products = pd.DataFrame({
    "product_id": product_ids,
    "product_name": product_names,
    "category": product_categories,
    "subcategory": subcategories,
    "price": prices
})

# Save dataset
products.to_csv(
    "data/raw/products.csv",
    index=False
)

print("Product dataset created successfully!")
print(f"Number of products: {len(products)}")

print("\nProduct categories:")
print(products["category"].value_counts())

print("\nFirst 5 products:")
print(products.head())