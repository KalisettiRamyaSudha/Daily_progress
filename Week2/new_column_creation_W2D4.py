"""
W2D4 - New Column Creation
Creates new columns based on calculations from existing columns,
such as total marks and percentage.
"""

import pandas as pd


def main():
    data = {
        "name": ["Asha", "Ravi", "Priya", "Kiran", "Meera"],
        "math": [85, 60, 95, 40, 72],
        "science": [90, 55, 92, 45, 68],
        "english": [78, 65, 98, 50, 75],
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:\n", df)

    # --- New column: total marks (sum of the three subject columns) ---
    df["total"] = df["math"] + df["science"] + df["english"]

    # --- New column: percentage (out of 300 total possible marks) ---
    df["percentage"] = (df["total"] / 300 * 100).round(2)

    # --- New column derived from another new column: pass/fail ---
    df["result"] = df["percentage"].apply(lambda pct: "Pass" if pct >= 40 else "Fail")

    print("\nWith total, percentage, and result columns added:\n", df)


if __name__ == "__main__":
    main()
