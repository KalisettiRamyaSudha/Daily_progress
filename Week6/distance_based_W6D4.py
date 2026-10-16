"""
Week 6 Day 4 - Distance-Based Outlier Check
Detect far-away points in a small numerical (2-feature) dataset using
simple distance-from-centroid logic.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)

# A tight cluster of "normal" points, plus 4 points planted far away
normal_points = rng.normal(loc=[50, 50], scale=[5, 5], size=(40, 2))
far_points = np.array([
    [90, 85],
    [10, 15],
    [95, 20],
    [5, 90],
])
X = np.vstack([normal_points, far_points])
df = pd.DataFrame(X, columns=["x", "y"])

# Distance-based logic: compute each point's Euclidean distance from the
# dataset's centroid (mean point), then flag anything unusually far.
centroid = df[["x", "y"]].mean().values
distances = np.sqrt(((df[["x", "y"]].values - centroid) ** 2).sum(axis=1))
df["distance_from_centroid"] = distances.round(2)

threshold = distances.mean() + 2 * distances.std()
df["is_outlier"] = df["distance_from_centroid"] > threshold

print(f"Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
print(f"Distance threshold (mean + 2*std): {threshold:.2f}")
print(f"\nFlagged {df['is_outlier'].sum()} outliers:")
print(df[df["is_outlier"]].sort_values("distance_from_centroid", ascending=False))

plt.figure(figsize=(7, 6))
colors = df["is_outlier"].map({True: "crimson", False: "steelblue"})
plt.scatter(df["x"], df["y"], c=colors, s=60, alpha=0.8, edgecolors="white")
plt.scatter(*centroid, c="black", marker="X", s=200, label="Centroid")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Distance-Based Outlier Detection")
plt.legend(["Centroid", "Normal", "Outlier"])
plt.tight_layout()
plt.savefig("distance_based_outlier_check_W6D4.png", dpi=150)
plt.show()

print(
    "\nThis is the same basic idea KMeans uses internally (distance to a "
    "center), just applied directly: a point far from where most of the "
    "data sits is flagged as unusual, without needing any labels."
)