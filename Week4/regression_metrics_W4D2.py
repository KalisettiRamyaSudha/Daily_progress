"""
W4D2 - MAE and MSE
Calculates Mean Absolute Error and Mean Squared Error to evaluate
regression performance.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"MAE  (Mean Absolute Error): {mae:.2f}")
print(f"MSE  (Mean Squared Error):  {mse:.2f}")
print(f"RMSE (Root MSE, same units as target): {rmse:.2f}")
print(f"R^2  (variance explained):  {r2:.3f}")

print(f"""
Interpretation:
- MAE ({mae:.2f}) is the average absolute size of the model's error, in the
  same units as the target - on average, predictions are off by about
  {mae:.1f} points.
- MSE ({mse:.2f}) squares each error before averaging, so it penalizes large
  mistakes much more heavily than small ones. It's harder to interpret
  directly since it's in squared units, which is why RMSE ({rmse:.2f}) is
  often reported instead - it's back in the original units.
- R^2 ({r2:.3f}) means the model explains about {r2:.0%} of the variance in
  the target. Closer to 1.0 is better; 0 would mean the model does no
  better than always predicting the average.
""")