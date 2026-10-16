"""
Week 6 Day 4 - Anomaly Detection Introduction
Use a basic z-score and IQR approach to detect unusual values in a
numeric column.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(10)

# Mostly normal transaction amounts, with a handful of clear anomalies
# (fraud-like spikes) mixed in
normal_values = rng.normal(50, 12, size=195)
anomalies = np.array([310, 275, 340, -40, 295])  # way outside the normal range
amounts = np.concatenate([normal_values, anomalies])
rng.shuffle(amounts)
df = pd.DataFrame({"transaction_amount": amounts.round(2)})

print(f"Dataset: {len(df)} transactions")
print(df["transaction_amount"].describe())

# --- Z-score method ---
# How many standard deviations each value sits from the mean. |z| > 3 is a
# common "unusual" threshold.
mean = df["transaction_amount"].mean()
std = df["transaction_amount"].std()
df["z_score"] = (df["transaction_amount"] - mean) / std
z_outliers = df[df["z_score"].abs() > 3]

print(f"\n=== Z-score method (|z| > 3) ===")
print(f"Flagged {len(z_outliers)} outliers:")
print(z_outliers[["transaction_amount", "z_score"]].sort_values("z_score"))

# --- IQR method ---
Q1 = df["transaction_amount"].quantile(0.25)
Q3 = df["transaction_amount"].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
iqr_outliers = df[(df["transaction_amount"] < lower) | (df["transaction_amount"] > upper)]

print(f"\n=== IQR method (outside [{lower:.2f}, {upper:.2f}]) ===")
print(f"Flagged {len(iqr_outliers)} outliers:")
print(iqr_outliers[["transaction_amount"]].sort_values("transaction_amount"))

z_set = set(z_outliers.index)
iqr_set = set(iqr_outliers.index)
only_iqr = iqr_set - z_set
only_z = z_set - iqr_set

print(
    f"\nZ-score flagged {len(z_outliers)}, IQR flagged {len(iqr_outliers)}. "
    f"They agree on {len(z_set & iqr_set)} points, but differ on the rest:\n"
    f"- IQR-only ({len(only_iqr)} points): milder values near the edge of "
    "the normal range that IQR's rank-based cutoff is more sensitive to.\n"
    f"- Z-score-only ({len(only_z)} points): none here, which hints at the "
    "real lesson - z-score actually MISSED the planted -40 anomaly. The "
    "five extreme high values inflate the mean and especially the std "
    "dev, which drags the z-score threshold wider and can mask a genuine "
    "anomaly on the opposite side of the distribution. IQR, being based on "
    "quartiles rather than mean/std, isn't thrown off the same way - one "
    "reason it's often preferred for data where outliers might distort "
    "the very statistics used to detect them."
)