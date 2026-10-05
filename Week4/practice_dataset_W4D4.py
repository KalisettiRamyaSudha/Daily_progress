"""
W4D4 - Practice Dataset
Repeats the full classification workflow (load -> split -> scale ->
train -> predict -> evaluate) end-to-end on the Iris dataset, as a
practice rep of everything from Days 1-3.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data
iris = load_iris(as_frame=True)
df = iris.frame
print("Dataset shape:", df.shape)
print("Classes:", [str(name) for name in iris.target_names])

# 2. Identify features and target
X = df.drop(columns=["target"])
y = df["target"]

# 3. Train-test split (stratified, since this is a 3-class problem)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 6. Predict
predictions = model.predict(X_test_scaled)

# 7. Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy:.1%})")

print("\nConfusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))