"""
Week 5 Day 4 - End-to-End Practice
One script that handles the complete machine learning workflow: raw data
in, trained and evaluated model out.
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

# --- 1. Load raw data ---
df = pd.read_csv("data/customer_churn_W5D4.csv")
print("Raw shape:", df.shape)
print("Missing values:\n", df.isna().sum())

# --- 2. Define features and target ---
X = df.drop(columns=["churn"])
y = df["churn"]
numeric_features = ["age", "income", "tenure_years", "monthly_charges"]
categorical_features = ["contract_type", "payment_method"]

# --- 3. Split before any preprocessing is fit, to avoid leakage ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Build preprocessing + model as one pipeline ---
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

# --- 5. Train ---
model.fit(X_train, y_train)

# --- 6. Predict ---
predictions = model.predict(X_test)

# --- 7. Evaluate ---
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("\n=== Evaluation ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("Confusion Matrix:\n", cm)

# --- 8. Sanity-check with cross-validation on the full pipeline ---
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"\n5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print(
    "\nConclusion: wrapping imputing, encoding, scaling, and the model into "
    "one Pipeline means every step - from raw CSV to prediction - happens "
    "in a single, repeatable call to .fit()/.predict(), with no manual "
    "bookkeeping and no risk of applying a transform inconsistently between "
    "training and test data."
)
