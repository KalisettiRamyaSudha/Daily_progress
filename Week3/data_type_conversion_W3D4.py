"""
W3D4 - Data Type Conversion
Converts strings to numeric values, parses dates, and adjusts category types.
"""

import pandas as pd

data = {
    "employee_id": ["101", "102", "103", "104"],
    "salary_str": ["72,000", "61,000", "68,500", "55,250"],
    "hire_date_str": ["2021-03-15", "2019-07-01", "2022-11-30", "2020-01-10"],
    "department": ["Engineering", "Sales", "Engineering", "HR"],
}
df = pd.DataFrame(data)
print("=== Original dtypes ===")
print(df.dtypes)
print(df)

# --- String -> numeric (strip commas, then convert) ---
df["employee_id"] = df["employee_id"].astype(int)
df["salary"] = df["salary_str"].str.replace(",", "", regex=False).astype(float)
df = df.drop(columns=["salary_str"])

# --- String -> datetime ---
df["hire_date"] = pd.to_datetime(df["hire_date_str"])
df = df.drop(columns=["hire_date_str"])

# --- String -> category (saves memory, signals a fixed set of values) ---
df["department"] = df["department"].astype("category")

print("\n=== After Conversion ===")
print(df)
print("\n=== New dtypes ===")
print(df.dtypes)

# A couple of things conversion now enables:
print("\nYears since hire (using datetime arithmetic):")
today = pd.Timestamp("2026-09-25")
df["years_since_hire"] = ((today - df["hire_date"]).dt.days / 365.25).round(1)
print(df[["employee_id", "hire_date", "years_since_hire"]])

print("\nAverage salary (now that it's numeric, not a string):", df["salary"].mean())
print("\nDepartment categories:", df["department"].cat.categories.tolist())