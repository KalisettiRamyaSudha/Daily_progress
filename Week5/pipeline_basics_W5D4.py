"""
Week 5 Day 4 - Pipeline Basics
Build a single sklearn Pipeline that combines preprocessing (imputing,
scaling, encoding via ColumnTransformer) and modeling into one object.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/customer_churn_W5D4.csv")
X = df.drop(columns=["churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = ["age", "income", "tenure_years", "monthly_charges"]
categorical_features = ["contract_type", "payment_method"]

# A ColumnTransformer applies a different preprocessing pipeline to
# different columns, then stitches the results back into one feature matrix.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

# The full pipeline: raw dataframe in, prediction out. Calling .fit() runs
# imputing -> scaling/encoding -> model training in one call; .predict()
# runs the same preprocessing on new data automatically, so there's no risk
# of forgetting a step or applying it inconsistently between train and test.
full_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])

full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Pipeline test accuracy: {accuracy:.4f}")
print("\nPipeline steps:")
for name, step in full_pipeline.named_steps.items():
    print(f"  {name}: {type(step).__name__}")
