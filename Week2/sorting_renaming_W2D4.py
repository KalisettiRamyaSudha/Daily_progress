"""
W2D4 - Sorting and Renaming
Renames columns and sorts records based on values in one or more columns.
"""

import pandas as pd


def main():
    data = {
        "name": ["Asha", "Ravi", "Priya", "Kiran", "Meera"],
        "age": [28, 34, 25, 41, 30],
        "dept": ["Engineering", "Sales", "Engineering", "HR", "Engineering"],
        "sal": [72000, 61000, 68000, 55000, 74000],
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:\n", df)

    # --- Renaming columns ---
    df_renamed = df.rename(columns={"dept": "department", "sal": "salary"})
    print("\nAfter renaming 'dept' -> 'department' and 'sal' -> 'salary':\n", df_renamed)

    # --- Sort by a single column (ascending) ---
    sorted_by_age = df_renamed.sort_values(by="age")
    print("\nSorted by age (ascending):\n", sorted_by_age)

    # --- Sort by a single column (descending) ---
    sorted_by_salary_desc = df_renamed.sort_values(by="salary", ascending=False)
    print("\nSorted by salary (descending):\n", sorted_by_salary_desc)

    # --- Sort by multiple columns ---
    sorted_multi = df_renamed.sort_values(by=["department", "salary"], ascending=[True, False])
    print("\nSorted by department (asc), then salary (desc) within each department:\n", sorted_multi)


if __name__ == "__main__":
    main()
