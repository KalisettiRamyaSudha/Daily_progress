"""
Week 5 Day 2 - Compare Tree vs. Forest
Compare decision tree and random forest metrics and summarize the differences.
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }

results_df = pd.DataFrame(results).T
print(results_df.round(4))

better = results_df["f1"].idxmax()
diff = results_df["f1"].max() - results_df["f1"].min()
print(f"\n{better} scores higher on F1 (by {diff:.4f}).")
print(
    "Summary: a single decision tree can fit the training data very closely, "
    "which makes it sensitive to noise in that particular split. A random "
    "forest averages many trees trained on different random subsets of the "
    "data and features, which smooths out that noise and typically improves "
    "generalization - usually at the cost of being slower to train and "
    "harder to interpret than one tree."
)