"""
Week 5 Day 4 - Categorical Encoding
Practice label encoding and one-hot encoding side by side on the same
categorical columns, and compare what each produces.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/customer_churn_W5D4.csv")
df["payment_method"] = df["payment_method"].fillna(df["payment_method"].mode()[0])

print("Original categorical columns:")
print(df[["contract_type", "payment_method"]].head())

# --- Label Encoding ---
# Assigns each category an integer (0, 1, 2, ...). Fine for a column with a
# real order (like contract_type: Month-to-month < One year < Two year), but
# risky for one without an order - the model could wrongly treat the numbers
# as meaning "greater than."
label_encoder = LabelEncoder()
df["contract_type_label_encoded"] = label_encoder.fit_transform(df["contract_type"])
print("\nLabel encoding mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"  {cls} -> {i}")

# --- One-Hot Encoding ---
# Creates one binary column per category. No ordering is implied, which is
# the safer default for a column like payment_method where the categories
# have no natural rank.
one_hot = pd.get_dummies(df["payment_method"], prefix="payment")
print("\nOne-hot encoded payment_method (first 5 rows):")
print(one_hot.head())

print(
    "\nWhen to use which: label encoding is compact (1 column) and fine for "
    "ordinal data or tree-based models that can split on arbitrary "
    "thresholds; one-hot encoding is safer for nominal (unordered) "
    "categories and for models like linear/logistic regression or KNN, "
    "where a label-encoded number could be misread as a magnitude."
)
