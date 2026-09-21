"""
W2D3 - CSV Read and Inspect
Creates a sample CSV file, loads it back, and inspects it using
head, tail, info, and describe.
"""

import pandas as pd
import os

FILENAME = "employees.csv"


def create_sample_csv(filename):
    data = {
        "name": ["John", "Dave", "Miya", "Kiran", "Meera", "Dev", "Nina"],
        "age": [28, 34, 25, 41, 30, 45, 22],
        "department": ["Engineering", "Sales", "Engineering", "HR",
                        "Engineering", "Sales", "HR"],
        "salary": [72000, 61000, 68000, 55000, 74000, 59000, 51000],
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df


def main():
    create_sample_csv(FILENAME)
    print(f"Created sample CSV: {FILENAME}")

    df = pd.read_csv(FILENAME)

    print("\n--- head() (first 5 rows) ---")
    print(df.head())

    print("\n--- tail(3) (last 3 rows) ---")
    print(df.tail(3))

    print("\n--- info() ---")
    df.info()

    print("\n--- describe() (summary statistics) ---")
    print(df.describe())

    # Clean up the file we created so repeated runs don't leave clutter
    os.remove(FILENAME)


if __name__ == "__main__":
    main()
