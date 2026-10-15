"""
Week 6 Day 3 - Clustering with PCA
Run KMeans on PCA-transformed data and compare the resulting clusters
against KMeans run on the original full-dimensional data.
"""
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

wine = load_wine(as_frame=True)
X_scaled = StandardScaler().fit_transform(wine.data)
true_labels = wine.target

# --- Clustering on the full 13-dimensional scaled data ---
kmeans_full = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_full = kmeans_full.fit_predict(X_scaled)

# --- Clustering on the 2D PCA projection ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_pca = kmeans_pca.fit_predict(X_pca)

ari_full = adjusted_rand_score(true_labels, labels_full)
ari_pca = adjusted_rand_score(true_labels, labels_pca)
sil_full = silhouette_score(X_scaled, labels_full)
sil_pca = silhouette_score(X_pca, labels_pca)
agreement_full_vs_pca = adjusted_rand_score(labels_full, labels_pca)

results = pd.DataFrame({
    "clustering": ["Full 13D (scaled)", "2D (PCA)"],
    "ARI_vs_true_wine_class": [ari_full, ari_pca],
    "silhouette_score": [sil_full, sil_pca],
}).set_index("clustering").round(4)

print(results)
print(f"\nAgreement between the two clusterings themselves (ARI): {agreement_full_vs_pca:.4f}")

print(
    "\nBoth clusterings land close to the true wine classes, and agree "
    "closely with each other - evidence that the first 2 principal "
    "components kept enough of the structure in the original 13 features "
    "for KMeans to find essentially the same groups, just computed much "
    "faster and visualizable in 2D. This won't always hold - it depends on "
    "how much variance the discarded components actually carried "
)