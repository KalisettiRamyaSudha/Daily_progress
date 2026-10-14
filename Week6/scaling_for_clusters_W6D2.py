"""
Week 6 Day 2 - Standardization Before Clustering
Compare KMeans clustering results before and after feature scaling, on
data where the "true" groups are separated mainly by a small-scale
feature (age) while a large-scale feature (income) varies randomly and
carries little group signal.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

rng = np.random.default_rng(3)
n = 200

# The TRUE grouping is driven by age (18-70) only. Income is random noise
# on a much larger numeric scale (20000-150000) with no real group signal -
# a stand-in for "a large-scale feature that happens to carry noise."
true_group = rng.choice([0, 1], size=n)
age = np.where(true_group == 0, rng.normal(25, 4, n), rng.normal(55, 4, n))
income = rng.normal(70000, 25000, n)  # same distribution regardless of group
df = pd.DataFrame({"age": age, "annual_income": income})

X_raw = df[["age", "annual_income"]].values
X_scaled = StandardScaler().fit_transform(X_raw)

kmeans_raw = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_raw = kmeans_raw.fit_predict(X_raw)

kmeans_scaled = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_scaled = kmeans_scaled.fit_predict(X_scaled)

# Adjusted Rand Index compares cluster assignments to the TRUE groups
# (1.0 = perfect match, 0.0 = no better than random)
ari_raw = adjusted_rand_score(true_group, labels_raw)
ari_scaled = adjusted_rand_score(true_group, labels_scaled)

print(f"Agreement with true age-based groups WITHOUT scaling (ARI): {ari_raw:.4f}")
print(f"Agreement with true age-based groups WITH scaling (ARI):    {ari_scaled:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(df["age"], df["annual_income"], c=labels_raw, cmap="viridis", s=30)
axes[0].set_title(f"Without Scaling (ARI vs. true groups={ari_raw:.3f})")
axes[0].set_xlabel("age")
axes[0].set_ylabel("annual_income")

axes[1].scatter(df["age"], df["annual_income"], c=labels_scaled, cmap="viridis", s=30)
axes[1].set_title(f"With Scaling (ARI vs. true groups={ari_scaled:.3f})")
axes[1].set_xlabel("age")
axes[1].set_ylabel("annual_income")

plt.tight_layout()
plt.savefig("scaling_for_clusters_W6D2.png", dpi=150)
plt.show()

print(
    "\nKMeans uses Euclidean distance, so a feature with much larger raw "
    "numbers (income, in the tens of thousands) dominates the distance "
    "calculation - even when, as here, it's pure noise with no real group "
    "signal. Unscaled, KMeans ends up clustering on income noise instead "
    "of the age pattern that actually defines the groups. Scaling first "
    "(StandardScaler) puts every feature on equal footing, so the "
    "clustering recovers the true age-based groups instead."
)
