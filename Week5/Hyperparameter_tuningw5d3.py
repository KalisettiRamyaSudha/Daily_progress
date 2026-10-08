"""
Week 5 Day 3 - Hyperparameter Tuning Introduction
Run a basic GridSearchCV example for one model (Random Forest).
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# The "grid" is every combination of these parameter values - 3 n_estimators
# x 3 max_depth x 2 min_samples_split = 18 combinations, each evaluated with
# 5-fold cross-validation (90 total model fits).
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [4, 8, None],
    "min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print(f"Combinations tested: {len(grid_search.cv_results_['params'])}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
print(f"Best parameters: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
test_preds = best_model.predict(X_test)
print(f"Test accuracy with best parameters: {accuracy_score(y_test, test_preds):.4f}")