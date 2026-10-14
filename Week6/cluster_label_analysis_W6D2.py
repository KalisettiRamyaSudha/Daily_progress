"""
Week 6 Day 2 - Cluster Label Analysis
Add cluster labels to the dataset and analyze what distinguishes each
cluster group.
"""
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=4, n_features=2, cluster_std=1.1, random_state=7)
df = pd.DataFrame(X, columns=["feature_1", "feature_2"])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

print("Dataset with cluster labels added:")
print(df.head())

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())

print("\nPer-cluster feature summary (mean, std):")
summary = df.groupby("cluster")[["feature_1", "feature_2"]].agg(["mean", "std"]).round(2)
print(summary)

print("\nPer-cluster analysis:")
for cluster_id in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == cluster_id]
    print(
        f"  Cluster {cluster_id}: {len(sub)} points, centered around "
        f"(feature_1={sub['feature_1'].mean():.2f}, feature_2={sub['feature_2'].mean():.2f}), "
        f"spread (std) = ({sub['feature_1'].std():.2f}, {sub['feature_2'].std():.2f})"
    )

print(
    "\nThis is the step that turns clustering from 'the algorithm found "
    "groups' into something useful: once every row has a cluster label, "
    "you can group by it like any other categorical column - compare "
    "means, look at size, and (on a real dataset) describe who or what "
    "each cluster actually represents."
)
