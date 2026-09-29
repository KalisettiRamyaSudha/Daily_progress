"""
W4D1 - Feature and Target Identification
Identifies independent variables (features, X) and the dependent
variable (target, y) in a machine learning dataset.
"""

import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame  # combined DataFrame: 4 feature columns + 1 target column

print("Full dataset columns:", list(df.columns))
print(df.head())

target_column = "target"
feature_columns = [col for col in df.columns if col != target_column]

X = df[feature_columns]
y = df[target_column]

print("\nIdentified feature columns (X):", feature_columns)
print("Identified target column (y):", target_column)
print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget is categorical (species: 0, 1, or 2) -> this is a classification problem.")
print("Unique target values:", sorted(y.unique()))