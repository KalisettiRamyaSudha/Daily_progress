"""
Week 5 Day 1 - Visualization of Results
Visualize the trained decision tree's structure and its feature importances.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Shallow depth here on purpose - a full-depth tree is unreadable as a diagram
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# --- Chart 1: the tree structure itself ---
plt.figure(figsize=(16, 8))
plot_tree(
    model, feature_names=X.columns, class_names=cancer.target_names,
    filled=True, rounded=True, fontsize=8,
)
plt.title("Decision Tree Structure (max_depth=3)")
plt.tight_layout()
plt.savefig("tree_structure_W5D1.png", dpi=150)
plt.show()

# --- Chart 2: feature importance ---
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top_features.sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Most Important Features")
plt.tight_layout()
plt.savefig("feature_importance_W5D1.png", dpi=150)
plt.show()

print("Top 5 features driving this tree's decisions:")
print(top_features.head())
