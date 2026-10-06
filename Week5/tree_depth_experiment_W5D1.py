"""
Week 5 Day 1 - Tree Depth Experiment
Change max_depth values and observe how training vs. test accuracy change -
a hands-on look at overfitting as a tree is allowed to grow deeper.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

depth_values = [1, 2, 3, 4, 5, 6, 8, 10, None]  # None = no limit (fully grown)
train_scores, test_scores = [], []

for depth in depth_values:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, model.predict(X_train)))
    test_scores.append(accuracy_score(y_test, model.predict(X_test)))

labels = [str(d) if d is not None else "None" for d in depth_values]
for label, tr, te in zip(labels, train_scores, test_scores):
    print(f"max_depth={label:>5} | train acc={tr:.4f} | test acc={te:.4f}")

plt.figure(figsize=(7, 5))
plt.plot(labels, train_scores, marker="o", label="Train Accuracy")
plt.plot(labels, test_scores, marker="o", label="Test Accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Accuracy vs. max_depth")
plt.legend()
plt.tight_layout()
plt.savefig("tree_depth_experiment_W5D1.png", dpi=150)
plt.show()

print(
    "\nObservation: as max_depth increases, train accuracy keeps climbing "
    "toward 1.0, but test accuracy levels off (or drops) once the tree "
    "starts memorizing noise in the training data rather than learning "
    "general patterns - that gap is overfitting."
)
