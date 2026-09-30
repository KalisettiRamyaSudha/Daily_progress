"""
W4D2 - Interpretation Notes
Trains the same linear regression model and prints notes explaining
what the coefficients mean, how predictions are formed, and how the
model is behaving overall.
"""

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
r2 = r2_score(y_test, model.predict(X_test))

coefficients = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)

print("=== Coefficients (sorted by magnitude) ===")
print(coefficients.round(1))
print(f"\nIntercept: {model.intercept_:.2f}")

print(f"""
=== Interpretation Notes ===

1. What the coefficients mean:
   Each coefficient tells you how much the prediction changes when that
   one feature increases by 1 unit, holding all other features constant.
   For example, '{coefficients.index[0]}' has the largest-magnitude
   coefficient ({coefficients.iloc[0]:+.1f}), meaning it has the strongest
   linear relationship with the target in this model - a {'positive' if coefficients.iloc[0] > 0 else 'negative'}
   coefficient means the target tends to {'increase' if coefficients.iloc[0] > 0 else 'decrease'}
   as that feature increases.

2. How a prediction is actually formed:
   prediction = intercept + (coef_1 * feature_1) + (coef_2 * feature_2) + ...
   The intercept ({model.intercept_:.2f}) is the model's baseline prediction
   when every feature is at its (scaled) average value; each term then
   nudges the prediction up or down from there.

3. Model behavior overall:
   This model explains about {r2:.0%} of the variance in the target
   (R^2 = {r2:.3f}). That leaves a meaningful amount unexplained, which
   makes sense - linear regression can only capture straight-line
   relationships, and real biological/medical measurements like these
   often interact in non-linear ways that this simple model can't see.

4. A caution about coefficient size:
   Coefficient magnitude only reflects "strength" fairly when features
   are on comparable scales. In this dataset the features are already
   mean-centered and scaled by sklearn, so comparing magnitudes directly
   is reasonable here - but with raw, unscaled features (e.g. one column
   in dollars, another in years), larger coefficients don't necessarily
   mean a stronger effect.
""")