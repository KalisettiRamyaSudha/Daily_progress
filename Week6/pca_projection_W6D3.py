"""
Week 6 Day 3 - 2D Projection
Transform higher-dimensional data (13 features) into two dimensions with
PCA and visualize it.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine(as_frame=True)
X = wine.data
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
# True class labels are only used here to color the plot for illustration -
# PCA itself never saw them, it's purely unsupervised
df_pca["wine_class"] = wine.target.map(dict(enumerate(wine.target_names)))

plt.figure(figsize=(7, 6))
for cls in df_pca["wine_class"].unique():
    subset = df_pca[df_pca["wine_class"] == cls]
    plt.scatter(subset["PC1"], subset["PC2"], label=cls, alpha=0.8, s=40)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Wine Dataset: 13 Features Projected to 2D via PCA")
plt.legend()
plt.tight_layout()
plt.savefig("pca_projection_W6D3.png", dpi=150)
plt.show()

print(
    "13 original features collapsed into just 2 axes, and the three wine "
    "classes still separate out visually - a sign that PC1 and PC2 "
    "together retain most of the information that actually distinguishes "
    "the classes, even though PCA was never told the classes existed "
    "(the color here is only added afterward, for us to read the plot)."
)