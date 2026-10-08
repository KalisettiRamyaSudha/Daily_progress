"""
Week 5 Day 3 - Best Parameters Analysis
Re-run the grid search, inspect the best parameters, and explain why they
likely improve performance over the defaults.
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

default_model = RandomForestClassifier(random_state=42)
default_model.fit(X_train, y_train)
default_acc = accuracy_score(y_test, default_model.predict(X_test))

best_model = grid_search.best_estimator_
best_acc = accuracy_score(y_test, best_model.predict(X_test))

print(f"Default RandomForestClassifier() test accuracy: {default_acc:.4f}")
print(f"Tuned (best params) test accuracy:               {best_acc:.4f}")
print(f"\nBest parameters found: {grid_search.best_params_}")

print(
    "\nWhy these parameters likely help:\n"
    f"- n_estimators={grid_search.best_params_['n_estimators']}: more trees "
    "means predictions are averaged over more independent random samples, "
    "which reduces variance - though returns shrink past a certain point.\n"
    f"- max_depth={grid_search.best_params_['max_depth']}: controls how "
    "deep each individual tree can grow; limiting it (vs. unlimited) can "
    "stop trees from memorizing noise, while too shallow a limit would "
    "underfit.\n"
    f"- min_samples_split={grid_search.best_params_['min_samples_split']}: "
    "the minimum rows needed at a node before it can split further - a "
    "higher value prevents splits based on very few, possibly noisy, samples.\n"
    "GridSearchCV picks the combination with the best mean cross-validation "
    "score, so this choice is backed by 5-fold validation rather than "
    "performance on a single lucky split."
)