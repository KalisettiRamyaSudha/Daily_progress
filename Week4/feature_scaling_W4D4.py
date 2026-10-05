"""
W4D4 - Feature Scaling Basics
Applies StandardScaler and MinMaxScaler and compares their effect on
model performance, using the breast cancer dataset (features on very
different scales).
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer(as_frame=True)
df = cancer.frame
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Feature ranges BEFORE scaling (first 3 columns):")
print(X_train.iloc[:, :3].describe().loc[["min", "max"]])

results = {}

# --- No scaling (baseline) ---
model_raw = KNeighborsClassifier(n_neighbors=5)
model_raw.fit(X_train, y_train)
results["No scaling"] = accuracy_score(y_test, model_raw.predict(X_test))

# --- StandardScaler: mean 0, std 1 ---
standard_scaler = StandardScaler()
X_train_std = standard_scaler.fit_transform(X_train)
X_test_std = standard_scaler.transform(X_test)
model_std = KNeighborsClassifier(n_neighbors=5)
model_std.fit(X_train_std, y_train)
results["StandardScaler"] = accuracy_score(y_test, model_std.predict(X_test_std))

# --- MinMaxScaler: squashes everything into [0, 1] ---
minmax_scaler = MinMaxScaler()
X_train_mm = minmax_scaler.fit_transform(X_train)
X_test_mm = minmax_scaler.transform(X_test)
model_mm = KNeighborsClassifier(n_neighbors=5)
model_mm.fit(X_train_mm, y_train)
results["MinMaxScaler"] = accuracy_score(y_test, model_mm.predict(X_test_mm))

print("\nKNN (K=5) accuracy by scaling method:")
for method, acc in results.items():
    print(f"  {method:<15} {acc:.4f} ({acc:.1%})")

print("""
What's different between the two scalers:
- StandardScaler centers each feature at mean 0 with standard deviation 1.
  It doesn't bound values to a fixed range, so outliers can still produce
  large values - but it handles data that's roughly normally distributed
  well.
- MinMaxScaler squashes every feature into a fixed [0, 1] range based on
  the min/max seen in training data. It's sensitive to outliers (a single
  extreme value compresses everything else into a narrow band), but it's
  useful when you need values in a bounded range.

Either one typically beats no scaling at all for distance-based models
like KNN, since unscaled features with larger raw ranges would otherwise
dominate the distance calculation regardless of their actual importance.
""")