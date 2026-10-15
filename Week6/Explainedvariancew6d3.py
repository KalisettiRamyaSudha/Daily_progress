"""
Week 6 Day 3 - Explained Variance
Inspect the explained variance ratio to understand how much information
PCA retains at each number of components.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine(as_frame=True)
X_scaled = StandardScaler().fit_transform(wine.data)

# Keep all 13 components so we can see the full picture of how much
# variance each one captures
pca = PCA(n_components=13)
pca.fit(X_scaled)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

summary = pd.DataFrame({
    "component": [f"PC{i+1}" for i in range(13)],
    "explained_variance_ratio": explained.round(4),
    "cumulative_variance": cumulative.round(4),
})
print(summary.to_string(index=False))

n_for_90 = int(np.argmax(cumulative >= 0.90) + 1)
print(f"\nComponents needed to retain >= 90% of the variance: {n_for_90}")
print(f"Variance retained by just the first 2 components: {cumulative[1]*100:.1f}%")

plt.figure(figsize=(8, 5))
plt.bar(summary["component"], summary["explained_variance_ratio"], alpha=0.7, label="Per-component")
plt.plot(summary["component"], summary["cumulative_variance"], color="red", marker="o", label="Cumulative")
plt.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="90% threshold")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Principal Component (Scree Plot)")
plt.legend()
plt.tight_layout()
plt.savefig("explained_variance_W6D3.png", dpi=150)
plt.show()

print(
    "\nEach bar is how much of the dataset's total variance that one "
    "component captures on its own; the red line is the running total. "
    "The first 2 components alone retain a majority of the information, "
    "which is why 2D PCA plots are usually a reasonable (if imperfect) "
    "summary of higher-dimensional data."
)