"""
Week 5 Day 2 - Overfitting Concept Practice
Compare training and testing accuracy across model complexity (n_estimators
and max_depth) to see overfitting directly in a random forest.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

depth_values = [1, 2, 3, 4, 5, 6, 8, 10, None]
train_scores, test_scores = [], []

for depth in depth_values:
    model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, model.predict(X_train)))
    test_scores.append(accuracy_score(y_test, model.predict(X_test)))

labels = [str(d) if d is not None else "None" for d in depth_values]
for label, tr, te in zip(labels, train_scores, test_scores):
    gap = tr - te
    print(f"max_depth={label:>5} | train acc={tr:.4f} | test acc={te:.4f} | gap={gap:.4f}")

plt.figure(figsize=(7, 5))
plt.plot(labels, train_scores, marker="o", label="Train Accuracy")
plt.plot(labels, test_scores, marker="o", label="Test Accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Random Forest: Train vs. Test Accuracy by max_depth")
plt.legend()
plt.tight_layout()
plt.savefig("overfitting_check_W5D2.png", dpi=150)
plt.show()

print(
    "\nObservation: the random forest's train/test gap grows much more "
    "slowly than a single decision tree's did on Day 1 - averaging 100 "
    "trees trained on random subsets reduces overfitting, even as individual "
    "trees inside the forest are allowed to grow deep."
)