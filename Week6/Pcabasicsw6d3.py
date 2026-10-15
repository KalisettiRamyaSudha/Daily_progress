"""
Week 6 Day 3 - PCA Basics
Learn dimensionality reduction basics and apply PCA to a dataset.
"""
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine(as_frame=True)
X = wine.data
print(f"Original data: {X.shape[0]} rows, {X.shape[1]} features")
print("Feature columns:", list(X.columns))

# PCA is sensitive to feature scale (like KMeans and KNN) - a feature with
# large raw numbers would dominate the first components otherwise - so
# scale first.
X_scaled = StandardScaler().fit_transform(X)

# PCA finds new axes (principal components), each a combination of the
# original 13 features, ordered so the first component captures the most
# variance (spread) in the data, the second captures the most of what's
# left, and so on.
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

print(f"\nReduced data: {X_pca.shape[0]} rows, {X_pca.shape[1]} components")
print("\nFirst 5 rows in the new component space:")
print(pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(5)]).head())

print(
    "\nDimensionality reduction, in short: 13 correlated chemical "
    "measurements (alcohol, acidity, color intensity, etc.) get compressed "
    "into 5 new uncorrelated components that summarize most of the "
    "original information - useful for visualization, speeding up models, "
    "and removing redundancy between correlated features."
)