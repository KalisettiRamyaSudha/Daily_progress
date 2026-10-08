"""
Week 5 Day 3 - Accuracy Comparison Sheet
Maintain a running table comparing every model built this week, with both
single-split and cross-validated accuracy.
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rows = []

# Logistic Regression (needs scaled features)
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train_scaled, y_train)
rows.append({
    "model": "Logistic Regression",
    "test_accuracy": accuracy_score(y_test, logreg.predict(X_test_scaled)),
    "cv_mean_accuracy": cross_val_score(LogisticRegression(max_iter=5000), X, y, cv=5).mean(),
})

# Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
rows.append({
    "model": "Decision Tree",
    "test_accuracy": accuracy_score(y_test, tree.predict(X_test)),
    "cv_mean_accuracy": cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=5).mean(),
})

# Random Forest (default)
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
rows.append({
    "model": "Random Forest (default)",
    "test_accuracy": accuracy_score(y_test, forest.predict(X_test)),
    "cv_mean_accuracy": cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=5).mean(),
})

# Random Forest (tuned via GridSearchCV)
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [4, 8, None], "min_samples_split": [2, 5]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
rows.append({
    "model": "Random Forest (tuned)",
    "test_accuracy": accuracy_score(y_test, grid_search.best_estimator_.predict(X_test)),
    "cv_mean_accuracy": grid_search.best_score_,
})

comparison_df = pd.DataFrame(rows).set_index("model").round(4)
comparison_df = comparison_df.sort_values("cv_mean_accuracy", ascending=False)

print(comparison_df)
comparison_df.to_csv("accuracy_comparison_W5D3.csv")
print("\nSaved to accuracy_comparison_W5D3.csv - keep appending a row here each time a new model is tried this week.")

best = comparison_df["cv_mean_accuracy"].idxmax()
print(f"\nBest model so far by cross-validated accuracy: {best}")