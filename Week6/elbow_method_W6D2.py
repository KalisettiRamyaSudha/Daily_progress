"""
Week 6 Day 2 - Elbow Method
Test different K values and use the elbow plot to choose a suitable
number of clusters.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Data actually has 4 underlying groups, but the elbow method pretends we
# don't know that - it's exactly the question we're trying to answer.
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, cluster_std=1.1, random_state=7)

k_values = range(1, 11)
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    print(f"k={k:>2} | inertia={kmeans.inertia_:.2f}")

plt.figure(figsize=(7, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("Elbow Method for Choosing k")
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig("elbow_method_W6D2.png", dpi=150)
plt.show()

print(
    "\nHow to read it: inertia always drops as k increases (more clusters "
    "can only fit the data at least as well), but the drop is steep at "
    "first and then flattens out. The 'elbow' - where adding another "
    "cluster stops buying much improvement - is the suggested k. Here the "
    "drop clearly levels off around k=4, which matches how this data was "
    "actually generated."
)
