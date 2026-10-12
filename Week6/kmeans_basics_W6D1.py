"""
Week 6 Day 1 - KMeans Basics
Apply KMeans clustering to sample data and inspect the resulting cluster
assignments.
"""
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=200, centers=3, n_features=2, cluster_std=1.2, random_state=42)
df = pd.DataFrame(X, columns=["feature_1", "feature_2"])

# KMeans groups points into k clusters by repeatedly: (1) assigning each
# point to its nearest cluster center, (2) moving each center to the mean
# of the points assigned to it, until the assignments stop changing.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

df["cluster"] = cluster_labels

print("First 10 rows with assigned cluster:")
print(df.head(10))

print("\nPoints per cluster:")
print(df["cluster"].value_counts().sort_index())

print("\nCluster centers (mean feature_1, feature_2 per cluster):")
print(pd.DataFrame(kmeans.cluster_centers_, columns=["feature_1", "feature_2"]))

print(f"\nInertia (sum of squared distances to nearest center): {kmeans.inertia_:.2f}")
print(
    "Lower inertia means points sit closer to their assigned cluster's "
    "center - tighter, more compact clusters."
)
