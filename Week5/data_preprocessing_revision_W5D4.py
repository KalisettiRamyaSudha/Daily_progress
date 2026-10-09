"""
Week 5 Day 4 - Data Preprocessing Revision
Repeat missing-value handling, encoding, scaling, and splitting in one
full workflow - a review of every preprocessing step covered so far.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("data/customer_churn_W5D4.csv")
print("Shape:", df.shape)
print("Missing values before cleaning:\n", df.isna().sum())

# --- Step 1: Missing-value handling ---
# Numeric columns: fill with the median (robust to outliers)
df["income"] = df["income"].fillna(df["income"].median())
df["tenure_years"] = df["tenure_years"].fillna(df["tenure_years"].median())
# Categorical column: fill with the most frequent category
df["payment_method"] = df["payment_method"].fillna(df["payment_method"].mode()[0])

print("\nMissing values after cleaning:\n", df.isna().sum().sum(), "total")

# --- Step 2: Encoding categorical columns ---
# Label-encode the target-like ordinal "contract_type" (has a natural order:
# Month-to-month < One year < Two year)
contract_order = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df["contract_type_encoded"] = df["contract_type"].map(contract_order)

# One-hot encode "payment_method" (no natural order between categories)
df = pd.get_dummies(df, columns=["payment_method"], prefix="payment")

# --- Step 3: Feature/target split ---
X = df.drop(columns=["churn", "contract_type"])
y = df["churn"]

# --- Step 4: Train/test split (before scaling, to avoid leakage) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 5: Scaling numeric columns (fit on train only) ---
numeric_cols = ["age", "income", "tenure_years", "monthly_charges"]
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(f"\nFinal feature columns ({len(X.columns)}):", list(X.columns))
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print("\nFirst 3 rows of fully preprocessed training data:")
print(X_train.head(3))
