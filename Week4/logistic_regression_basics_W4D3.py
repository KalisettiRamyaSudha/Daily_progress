"""
W4D3 - Logistic Regression Basics
Builds a logistic regression model using a binary classification
dataset (sklearn's breast cancer dataset: malignant vs. benign).
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer(as_frame=True)
df = cancer.frame

X = df.drop(columns=["target"])
y = df["target"]

print("Target classes:", {i: str(name) for i, name in enumerate(cancer.target_names)})
print("Class balance:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic regression benefits from scaled features (many features here
# are on very different scales, e.g. "mean area" vs "mean smoothness")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]  # P(class=1)

print(f"\nTraining set size: {len(X_train)} | Test set size: {len(X_test)}")
print("\nFirst 10 predictions (with predicted probability of class 1):")
for pred, prob, actual in list(zip(predictions, probabilities, y_test))[:10]:
    print(f"  predicted={pred}  P(1)={prob:.2f}  actual={actual}")