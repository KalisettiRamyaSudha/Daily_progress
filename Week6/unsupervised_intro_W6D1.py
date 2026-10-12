"""
Week 6 Day 1 - Unsupervised Learning Introduction
Review the idea of clustering: grouping data by similarity with no labels
to learn from, as opposed to the supervised workflows used in Weeks 4-5.
"""
import pandas as pd
from sklearn.datasets import make_blobs

# make_blobs generates points clustered around a few random centers. We'll
# keep the true cluster labels aside in y_true ONLY to describe the data
# honestly below - a real unsupervised problem would never have this.
X, y_true = make_blobs(n_samples=200, centers=3, n_features=2, cluster_std=1.2, random_state=42)
df = pd.DataFrame(X, columns=["feature_1", "feature_2"])

print("Shape of X (features only - no target column):", df.shape)
print(df.head())

print(
    "\nSupervised vs. unsupervised, in one dataset:\n"
    "- Supervised (Weeks 4-5): every row had a known answer (y) - species, "
    "diagnosis, pass/fail - and the model learned to predict that answer.\n"
    "- Unsupervised (this week): there is no y at all. The algorithm only "
    "sees feature_1 and feature_2, and has to find structure - groups of "
    "similar points - on its own, with no 'correct' labels to check against."
)

print(
    f"\n(For reference only: this synthetic data was actually generated from "
    f"{len(set(y_true))} underlying groups - but a clustering algorithm "
    f"never gets to see that; the next two scripts show KMeans discovering "
    f"structure in these same points without being told how many groups "
    f"exist or which point belongs to which.)"
)
