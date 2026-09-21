"""
W2D4 - Grouping Basics
Uses groupby to calculate counts and averages by category.
"""

import pandas as pd


def main():
    data = {
        "name": ["Asha", "Ravi", "Priya", "Kiran", "Meera", "Dev", "Nina"],
        "department": ["Engineering", "Sales", "Engineering", "HR",
                        "Engineering", "Sales", "HR"],
        "salary": [72000, 61000, 68000, 55000, 74000, 59000, 51000],
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:\n", df)

    # --- Count of employees per department ---
    dept_counts = df.groupby("department").size()
    print("\nEmployee count per department:\n", dept_counts)

    # --- Average salary per department ---
    avg_salary = df.groupby("department")["salary"].mean()
    print("\nAverage salary per department:\n", avg_salary)

    # --- Multiple aggregations at once ---
    summary = df.groupby("department")["salary"].agg(["count", "mean", "min", "max"])
    print("\nFull salary summary per department:\n", summary)


if __name__ == "__main__":
    main()
