"""
Week 6 Day 1 - Cluster Visualization
Create a scatter plot that clearly shows the clusters KMeans found on a
two-feature dataset, including the cluster centers.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=200, centers=3, n_features=2, cluster_std=1.2, random_state=42)
df = pd.DataFrame(X, columns=["feature_1", "feature_2"])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

plt.figure(figsize=(7, 6))
scatter = plt.scatter(
    df["feature_1"], df["feature_2"],
    c=df["cluster"], cmap="viridis", s=40, alpha=0.8, edgecolors="white", linewidths=0.5,
)
plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    c="red", marker="X", s=250, edgecolors="black", linewidths=1.5, label="Cluster centers",
)
plt.xlabel("feature_1")
plt.ylabel("feature_2")
plt.title("KMeans Clustering (k=3)")
plt.legend()
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("cluster_visualization_W6D1.png", dpi=150)
# plt.show()

print("Chart saved: cluster_visualization_W6D1.png")
print(
    "Each color is a cluster KMeans discovered on its own, with no labels "
    "given; the red X markers show where each cluster's center landed. "
    "Well-separated colors with little overlap between them is a visual "
    "sign the clustering fits this data well."
)
