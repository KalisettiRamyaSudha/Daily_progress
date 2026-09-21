"""
W2D5 - Mini Dataset Project
Creates a small sales dataset, cleans it (handles missing/duplicate
data), analyzes it (aggregations, groupby), and prints a summary
report. Pulls together the Week 2 NumPy/pandas topics: array math,
DataFrame creation, missing values, new columns, sorting, and groupby.
"""

import pandas as pd
import numpy as np

pd.set_option("display.width", 100)


def create_dataset():
    """Builds a small, intentionally messy sales dataset."""
    data = {
        "order_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "product": ["Laptop", "Mouse", "Keyboard", "Monitor", "Laptop",
                    "Mouse", "Monitor", "Keyboard", "Laptop", "Mouse"],
        "region": ["East", "West", "East", "South", "West",
                   "East", "West", "South", "East", np.nan],
        "quantity": [2, 5, 3, np.nan, 1, 4, 2, 3, 1, 6],
        "unit_price": [950.00, 22.50, 45.00, 210.00, 950.00,
                       22.50, 210.00, np.nan, 950.00, 22.50],
    }
    return pd.DataFrame(data)


def clean_dataset(df):
    """Fills missing values and removes exact duplicate rows."""
    df = df.copy()

    # Region: unknown region -> "Unspecified"
    df["region"] = df["region"].fillna("Unspecified")

    # Quantity: fill with the median quantity (a reasonable typical order size)
    df["quantity"] = df["quantity"].fillna(df["quantity"].median())

    # Unit price: fill missing prices using the product's known price elsewhere
    df["unit_price"] = df.groupby("product")["unit_price"].transform(
        lambda prices: prices.fillna(prices.median())
    )

    df = df.drop_duplicates()
    return df


def analyze_dataset(df):
    """Adds a revenue column and computes summary statistics."""
    df = df.copy()
    df["revenue"] = df["quantity"] * df["unit_price"]
    return df


def print_summary(df):
    print("=== Cleaned & Analyzed Dataset ===")
    print(df)

    print("\n=== Revenue by Product ===")
    print(df.groupby("product")["revenue"].sum().sort_values(ascending=False))

    print("\n=== Revenue by Region ===")
    print(df.groupby("region")["revenue"].sum().sort_values(ascending=False))

    print("\n=== Overall Summary ===")
    print(f"Total orders:      {len(df)}")
    print(f"Total units sold:  {int(df['quantity'].sum())}")
    print(f"Total revenue:     ${df['revenue'].sum():,.2f}")
    print(f"Average order value: ${df['revenue'].mean():,.2f}")
    top_product = df.groupby("product")["revenue"].sum().idxmax()
    print(f"Best-selling product (by revenue): {top_product}")


def main():
    raw_df = create_dataset()
    print("=== Raw Dataset (with missing values) ===")
    print(raw_df)
    print("\nMissing values per column:\n", raw_df.isna().sum())

    cleaned_df = clean_dataset(raw_df)
    analyzed_df = analyze_dataset(cleaned_df)

    print()
    print_summary(analyzed_df)


if __name__ == "__main__":
    main()
