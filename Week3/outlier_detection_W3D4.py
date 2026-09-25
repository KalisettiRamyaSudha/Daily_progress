"""
W3D4 - Outlier Detection Basics
Uses box plots and the IQR (Interquartile Range) method to detect outliers.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=5)
salaries = rng.normal(65000, 8000, size=50).round(0)
# Inject a few clear outliers
salaries = np.append(salaries, [150000, 12000, 145000])
df = pd.DataFrame({"salary": salaries})

# --- IQR method ---
Q1 = df["salary"].quantile(0.25)
Q3 = df["salary"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: {Q1:.2f}")
print(f"Q3: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")

outliers = df[(df["salary"] < lower_bound) | (df["salary"] > upper_bound)]
print(f"\nOutliers found ({len(outliers)}):")
print(outliers)

non_outliers = df[(df["salary"] >= lower_bound) & (df["salary"] <= upper_bound)]
print(f"\nRemaining rows after excluding outliers: {len(non_outliers)}")

# --- Box plot visualization ---
plt.figure(figsize=(7, 5))
plt.boxplot(df["salary"], vert=True, patch_artist=True,
            boxprops=dict(facecolor="#a6cee3"),
            flierprops=dict(markerfacecolor="red", marker="o", markersize=8))
plt.title("Salary Distribution with Outliers (Box Plot)")
plt.ylabel("Salary ($)")
plt.tight_layout()
plt.savefig("outlier_boxplot_W3D4.png", dpi=150)
print("\nSaved outlier_boxplot_W3D4.png")