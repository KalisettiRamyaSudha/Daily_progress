"""
W2D3 - Column Selection
Selects single and multiple columns from a DataFrame and filters
rows using conditions.
"""

import pandas as pd


def main():
    data = {
        "name": ["John", "Dave", "Miya", "Kiran", "Meera"],
        "age": [28, 34, 25, 41, 30],
        "department": ["Engineering", "Sales", "Engineering", "HR", "Engineering"],
        "salary": [72000, 61000, 68000, 55000, 74000],
    }
    df = pd.DataFrame(data)
    print("Full DataFrame:\n", df)

    # --- Single column selection ---
    print("\nSingle column (df['name']):\n", df["name"])

    # --- Multiple column selection ---
    print("\nMultiple columns (df[['name', 'salary']]):\n", df[["name", "salary"]])

    # --- Filtering rows with a single condition ---
    print("\nEmployees older than 28 (df[df['age'] > 28]):\n", df[df["age"] > 28])

    # --- Filtering rows with multiple conditions ---
    engineering_high_earners = df[(df["department"] == "Engineering") & (df["salary"] > 65000)]
    print("\nEngineering employees earning > 65000:\n", engineering_high_earners)

    # --- Selecting specific columns from filtered rows ---
    print("\nNames of employees over 30:\n", df[df["age"] > 30]["name"])


if __name__ == "__main__":
    main()
