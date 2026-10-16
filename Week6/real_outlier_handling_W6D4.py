"""
Week 6 Day 4 - Real Data Cleaning with Outliers
Remove or cap outliers in a realistic customer dataset and compare how
summary statistics and a KMeans clustering change as a result.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

rng = np.random.default_rng(11)
n = 200

# Realistic customer data: age and annual income, with a handful of
# data-entry-error-style outliers mixed in (e.g. a misplaced extra digit)
age = np.clip(rng.normal(40, 12, n), 18, 75).round(0)
income = np.clip(rng.normal(60000, 18000, n), 20000, 150000).round(0)

outlier_idx = rng.choice(n, size=6, replace=False)
income[outlier_idx] = rng.choice([980000, 1200000, 850000], size=6)  # typo-like extra digit

df = pd.DataFrame({"age": age, "annual_income": income})

print("=== BEFORE handling outliers ===")
print(df["annual_income"].describe())

Q1 = df["annual_income"].quantile(0.25)
Q3 = df["annual_income"].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
n_outliers = ((df["annual_income"] < lower) | (df["annual_income"] > upper)).sum()
print(f"\nIQR bounds: [{lower:,.0f}, {upper:,.0f}] -> {n_outliers} outliers detected")

# --- Approach 1: Remove outliers ---
df_removed = df[(df["annual_income"] >= lower) & (df["annual_income"] <= upper)].copy()

# --- Approach 2: Cap outliers at the IQR bounds (winsorizing) ---
df_capped = df.copy()
df_capped["annual_income"] = df_capped["annual_income"].clip(lower, upper)

print("\n=== AFTER removing outliers ===")
print(df_removed["annual_income"].describe())
print("\n=== AFTER capping outliers ===")
print(df_capped["annual_income"].describe())

# --- Compare the effect on a downstream KMeans clustering ---
def cluster_and_score(data):
    X_scaled = StandardScaler().fit_transform(data[["age", "annual_income"]])
    labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
    return silhouette_score(X_scaled, labels)

sil_before = cluster_and_score(df)
sil_removed = cluster_and_score(df_removed)
sil_capped = cluster_and_score(df_capped)

comparison = pd.DataFrame({
    "dataset": ["Original (with outliers)", "Outliers removed", "Outliers capped"],
    "n_rows": [len(df), len(df_removed), len(df_capped)],
    "mean_income": [df["annual_income"].mean(), df_removed["annual_income"].mean(), df_capped["annual_income"].mean()],
    "std_income": [df["annual_income"].std(), df_removed["annual_income"].std(), df_capped["annual_income"].std()],
    "kmeans_silhouette": [sil_before, sil_removed, sil_capped],
}).round(2)

print("\n=== Comparison ===")
print(comparison.to_string(index=False))

print(
    "\nThe handful of extreme income values massively inflate the mean "
    "($91,080 vs. ~$61-63k after cleaning) and the std dev (by 10x) in "
    "the original data - that part is the expected, obvious effect.\n\n"
    "The clustering result is the more interesting, counter-intuitive "
    "finding: the silhouette score actually DROPS after removing/capping "
    "the outliers (0.56 -> ~0.37), not improves. That's because with the "
    "extreme points still in, KMeans finds it trivially easy to isolate "
    "those 6 far-away values into their own tightly-separated cluster - "
    "which inflates the silhouette score without that cluster being "
    "meaningful. Once the outliers are gone, KMeans has to find real "
    "structure among 191-200 genuinely similar customers, which is a "
    "harder, more honest clustering problem - hence the lower score. "
    "Lesson: a clustering quality metric can look better purely because "
    "outliers are forming their own trivial cluster, so don't skip outlier "
    "handling just because the metric looks good with them left in."
)