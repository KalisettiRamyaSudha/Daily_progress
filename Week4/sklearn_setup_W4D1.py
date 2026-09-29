from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data and identify features/target
iris = load_iris(as_frame=True)
df = iris.frame
X = df.drop(columns=["target"])
y = df["target"]

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Instantiate the model (classification, since target is categorical)
model = LogisticRegression(max_iter=200)

# 4. Fit - the model learns only from the training data
model.fit(X_train, y_train)

# 5. Predict - generate predictions for the held-out test features
predictions = model.predict(X_test)

# 6. Evaluate - compare predictions to the true test labels
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on test set: {accuracy:.2%}")
print("\nSample predictions vs actual:")
for pred, actual in list(zip(predictions, y_test))[:10]:
    match = "correct" if pred == actual else "WRONG"
    print(f"  predicted={pred}, actual={actual}  [{match}]")

print("\nFull classification report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))