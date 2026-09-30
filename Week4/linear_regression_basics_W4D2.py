"""
W4D2 - Linear Regression Basics
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Feature columns:", list(X.columns))
print(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")

print("\nFirst 10 predictions vs actual values:")
for pred, actual in list(zip(predictions, y_test))[:10]:
    print(f"  predicted={pred:7.1f}   actual={actual:7.1f}   error={pred - actual:+7.1f}")