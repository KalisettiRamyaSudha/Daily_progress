"""
Week 6 Day 2 - Real Dataset Clustering
Apply clustering to a small, realistic customer dataset (age, annual
income, spending score - the classic "mall customer segmentation" style
problem) and interpret the resulting segments.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(99)
n = 200

# Five realistic customer segments: young low spenders, young high
# spenders, mid-age average spenders, older high earners/low spenders,
# and older moderate spenders
segment_params = [
    {"age": (22, 4), "income": (25, 6), "spending": (75, 10), "n": 40},   # young, high spending
    {"age": (25, 4), "income": (22, 5), "spending": (20, 10), "n": 40},   # young, low spending
    {"age": (40, 6), "income": (55, 10), "spending": (50, 12), "n": 40},  # mid-age, average
    {"age": (55, 6), "income": (85, 12), "spending": (20, 10), "n": 40},  # older, high income, low spend
    {"age": (50, 6), "income": (45, 8), "spending": (55, 10), "n": 40},   # older, moderate
]

rows = []
for params in segment_params:
    age = rng.normal(*params["age"], params["n"])
    income = rng.normal(*params["income"], params["n"])
    spending = np.clip(rng.normal(*params["spending"], params["n"]), 1, 100)
    rows.append(pd.DataFrame({"age": age, "annual_income_k": income, "spending_score": spending}))

df = pd.concat(rows, ignore_index=True)
df["age"] = df["age"].clip(18, 70).round(0)
df["annual_income_k"] = df["annual_income_k"].clip(15, 120).round(1)
df["spending_score"] = df["spending_score"].round(1)

print("Dataset shape:", df.shape)
print(df.head())

X = df[["age", "annual_income_k", "spending_score"]]
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster profiles (mean values):")
profile = df.groupby("cluster")[["age", "annual_income_k", "spending_score"]].mean().round(1)
profile["count"] = df["cluster"].value_counts().sort_index()
print(profile)

fig, ax = plt.subplots(figsize=(7, 6))
scatter = ax.scatter(
    df["annual_income_k"], df["spending_score"], c=df["cluster"], cmap="tab10", s=40, alpha=0.8
)
ax.set_xlabel("Annual Income (k)")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segments (colored by cluster)")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("real_dataset_clustering_W6D2.png", dpi=150)
plt.show()

print(
    "\nInterpretation: each cluster is a customer segment a business could "
    "target differently - e.g. a high-income, low-spending cluster might "
    "respond to a different marketing approach than a young, high-spending "
    "cluster, even though neither was ever told these labels existed - "
    "KMeans found the grouping purely from the three numeric features."
)
