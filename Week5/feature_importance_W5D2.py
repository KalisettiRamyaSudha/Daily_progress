"""
Week 5 Day 2 - Feature Importance
Extract and visualize the most important features from the trained
random forest.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance here is the average reduction in impurity each feature
# contributes, averaged across all 100 trees in the forest.
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top_features.sort_values().plot(kind="barh", color="#4c72b0")
plt.xlabel("Importance")
plt.title("Top 10 Most Important Features (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance_W5D2.png", dpi=150)
plt.show()

print("Top 10 features driving this forest's decisions:")
print(top_features)