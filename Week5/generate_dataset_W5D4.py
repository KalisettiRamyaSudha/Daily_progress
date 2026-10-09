"""
Generates data/customer_churn_W5D4.csv - a small synthetic customer dataset
with a mix of numeric and categorical columns, plus a few missing values,
so Thursday's preprocessing tasks have real cleaning work to do.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=7)
n = 400

age = rng.integers(18, 75, size=n)
tenure_years = np.clip(rng.normal(4, 2.5, size=n), 0, 20).round(1)
monthly_charges = np.clip(rng.normal(70, 25, size=n), 20, 150).round(2)

contract_type = rng.choice(
    ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
)
payment_method = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n
)

# Churn probability depends on contract type, tenure, and monthly charges,
# so the target has a real, learnable signal (not just noise)
base_churn_prob = np.select(
    [contract_type == "Month-to-month", contract_type == "One year"],
    [0.55, 0.20],
    default=0.05,
)
charge_effect = np.clip((monthly_charges - 70) / 150, -0.2, 0.3)
churn_prob = np.clip(base_churn_prob - tenure_years * 0.035 + charge_effect, 0.03, 0.95)
churn = (rng.random(n) < churn_prob).astype(int)

income = np.clip(rng.normal(55000, 18000, size=n), 18000, 150000).round(0)

df = pd.DataFrame({
    "age": age,
    "income": income,
    "tenure_years": tenure_years,
    "monthly_charges": monthly_charges,
    "contract_type": contract_type,
    "payment_method": payment_method,
    "churn": churn,
})

# Introduce missing values, like a real export would have
missing_income_idx = rng.choice(n, size=18, replace=False)
df.loc[missing_income_idx, "income"] = np.nan
missing_tenure_idx = rng.choice(n, size=12, replace=False)
df.loc[missing_tenure_idx, "tenure_years"] = np.nan
missing_payment_idx = rng.choice(n, size=10, replace=False)
df.loc[missing_payment_idx, "payment_method"] = np.nan

df.to_csv("data/customer_churn_W5D4.csv", index=False)
print(f"Wrote data/customer_churn_W5D4.csv with {len(df)} rows")
print(df["churn"].value_counts())
print("\nMissing values:\n", df.isna().sum())
