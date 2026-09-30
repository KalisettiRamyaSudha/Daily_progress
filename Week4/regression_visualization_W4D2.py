"""
W4D2 - Regression Visualization
Plots actual vs. predicted values for a regression model's output.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, predictions, alpha=0.6, color="#4c72b0", edgecolor="black")

# Perfect-prediction reference line: if every point sat exactly on this
# line, predicted would always equal actual
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--",
         linewidth=2, label="Perfect prediction (y = x)")

plt.title("Actual vs. Predicted Values (Linear Regression)")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("regression_actual_vs_predicted_W4D2.png", dpi=150)
print("Saved regression_actual_vs_predicted_W4D2.png")
print("\nPoints closer to the red dashed line indicate more accurate predictions;")
print("points far above/below it are cases the model over/under-predicted.")