"""
W3D4 - Data Cleaning Practice
Identifies and removes duplicate rows, and standardizes inconsistent
column formats (whitespace, casing, spelling variants).
"""

import pandas as pd

data = {
    "name": [" Asha ", "Ravi", "priya", "Kiran", "Asha", "RAVI", "Meera"],
    "department": ["engineering", "Sales ", "Engineering", "HR", "engineering", "sales", "Hr"],
    "salary": [72000, 61000, 68000, 55000, 72000, 61000, 74000],
}
df = pd.DataFrame(data)
print("=== Raw Data ===")
print(df)

# --- Standardize column formats first, so duplicates hidden by
#     inconsistent casing/whitespace become detectable ---
df["name"] = df["name"].str.strip().str.title()
df["department"] = df["department"].str.strip().str.title()

print("\n=== After Standardizing Text Formats ===")
print(df)

# --- Identify duplicates (now that formatting is consistent) ---
duplicate_mask = df.duplicated(subset=["name", "department", "salary"], keep="first")
print(f"\nDuplicate rows found: {duplicate_mask.sum()}")
print(df[duplicate_mask])

# --- Remove duplicates ---
cleaned_df = df.drop_duplicates(subset=["name", "department", "salary"], keep="first").reset_index(drop=True)
print("\n=== After Removing Duplicates ===")
print(cleaned_df)

print(f"\nRows before cleaning: {len(df)}")
print(f"Rows after cleaning:  {len(cleaned_df)}")