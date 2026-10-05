"""
W4D4 - Model Comparison
Compares logistic regression and KNN on the same dataset and reports
which performs better and why.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

cancer = load_breast_cancer(as_frame=True)
df = cancer.frame
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    results[name] = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }

print(f"{'Model':<22}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<10}")
print("-" * 68)
for name, metrics in results.items():
    print(f"{name:<22}{metrics['accuracy']:<12.4f}{metrics['precision']:<12.4f}"
          f"{metrics['recall']:<12.4f}{metrics['f1']:<10.4f}")

best_model = max(results, key=lambda name: results[name]["f1"])
print(f"\nBest model by F1-score: {best_model}")

print("""
Why one might outperform the other here:
- Logistic Regression assumes a roughly linear decision boundary between
  classes (in terms of the scaled features). If the real boundary between
  malignant/benign is close to linear in this feature space, it can match
  or beat KNN while being simpler and faster to train.
- KNN makes no assumption about the boundary's shape - it just looks at
  nearby points - which helps when the true boundary is curved or
  irregular, but it can be more sensitive to noisy/overlapping points
  near the boundary and to the choice of K.
- With a clean, well-separated, moderately-sized dataset like this one,
  the two often land close together in accuracy; the gap matters more as
  the data gets messier, more non-linear, or higher-dimensional.

In practice, "which model is better" isn't answered once - it's worth
comparing a couple of reasonable candidates on your actual data (as done
here) rather than assuming one algorithm is always superior.
""")