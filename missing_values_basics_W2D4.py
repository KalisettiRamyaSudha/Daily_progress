"""
W2D4 - Missing Values Basics
Creates missing values in a dataset and practices identifying,
filling, and dropping them.
"""

import pandas as pd
import numpy as np


def main():
    data = {
        "name": ["Asha", "Ravi", "Priya", "Kiran", "Meera", "Dev"],
        "age": [28, 34, np.nan, 41, 30, np.nan],
        "department": ["Engineering", "Sales", "Engineering", None, "Engineering", "Sales"],
        "salary": [72000, np.nan, 68000, 55000, 74000, np.nan],
    }
    df = pd.DataFrame(data)
    print("DataFrame with missing values:\n", df)

    # --- Identifying missing values ---
    print("\nMissing value map (isna()):\n", df.isna())
    print("\nMissing value count per column:\n", df.isna().sum())
    print("\nTotal missing values in the DataFrame:", df.isna().sum().sum())

    # --- Filling missing values ---
    df_filled = df.copy()
    df_filled["age"] = df_filled["age"].fillna(df_filled["age"].mean())
    df_filled["salary"] = df_filled["salary"].fillna(df_filled["salary"].median())
    df_filled["department"] = df_filled["department"].fillna("Unknown")
    print("\nAfter filling (age -> mean, salary -> median, department -> 'Unknown'):\n", df_filled)

    # --- Dropping missing values ---
    df_dropped_rows = df.dropna()
    print("\nAfter dropping any row with a missing value:\n", df_dropped_rows)

    df_dropped_cols = df.dropna(axis=1)
    print("\nAfter dropping any column with a missing value:\n", df_dropped_cols)

    # dropna with a threshold: keep rows with at least 3 non-null values
    df_dropped_thresh = df.dropna(thresh=3)
    print("\nAfter dropping rows with fewer than 3 non-null values:\n", df_dropped_thresh)


if __name__ == "__main__":
    main()
