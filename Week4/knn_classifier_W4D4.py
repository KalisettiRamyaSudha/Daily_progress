"""
W4D4 - KNN Classifier
Builds a K-Nearest Neighbors classifier and experiments with different
K values, using the Iris dataset.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris(as_frame=True)
df = iris.frame
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# KNN is distance-based, so scaling matters even more here than for
# logistic regression - an unscaled feature with a larger numeric range
# would dominate the distance calculation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
accuracies = []

print("K value -> Test accuracy")
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    accuracies.append(acc)
    print(f"  K={k:<3} -> {acc:.4f} ({acc:.1%})")

best_k = k_values[accuracies.index(max(accuracies))]
print(f"\nBest K on this test set: {best_k} (accuracy = {max(accuracies):.1%})")

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker="o", color="#4c72b0")
plt.title("KNN Accuracy vs. K Value")
plt.xlabel("K (number of neighbors)")
plt.ylabel("Test Accuracy")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("knn_accuracy_vs_k_W4D4.png", dpi=150)
print("Saved knn_accuracy_vs_k_W4D4.png")

print("""
Notes on choosing K:
- Very small K (e.g. 1) can overfit - it follows noise in individual
  training points very closely.
- Very large K can underfit - it smooths over real class boundaries by
  averaging across too many neighbors, some of which may belong to a
  different class.
- A common practice is to try a range of odd K values (odd avoids tie
  votes in binary classification) and pick the one with the best
  validation/test performance, as this script does.
""")