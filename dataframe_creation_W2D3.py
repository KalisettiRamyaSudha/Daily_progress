"""
W2D3 - DataFrame Creation
Creates DataFrames from dictionaries and lists, and inspects
rows and columns.
"""

import pandas as pd


def main():
    # --- DataFrame from a dictionary of lists ---
    data_dict = {
        "name": ["Asha", "Ravi", "Priya", "Kiran"],
        "age": [28, 34, 25, 41],
        "department": ["Engineering", "Sales", "Engineering", "HR"],
    }
    df_from_dict = pd.DataFrame(data_dict)
    print("DataFrame from a dictionary:\n", df_from_dict)

    # --- DataFrame from a list of lists (with explicit column names) ---
    data_list = [
        ["Laptop", 999.99, 5],
        ["Mouse", 25.50, 40],
        ["Keyboard", 75.00, 15],
    ]
    df_from_list = pd.DataFrame(data_list, columns=["product", "price", "stock"])
    print("\nDataFrame from a list of lists:\n", df_from_list)

    # --- Inspecting rows ---
    print("\nFirst 2 rows (df_from_dict.head(2)):\n", df_from_dict.head(2))
    print("\nRow at index 1 (df_from_dict.loc[1]):\n", df_from_dict.loc[1])
    print("\nRow shape (rows, columns):", df_from_dict.shape)

    # --- Inspecting columns ---
    print("\nColumn names:", list(df_from_dict.columns))
    print("Just the 'name' column:\n", df_from_dict["name"])
    print("\nData types per column:\n", df_from_dict.dtypes)


if __name__ == "__main__":
    main()
