"""
W4D3 - Classification Report
Uses precision, recall, and F1-score to evaluate the classification
model beyond plain accuracy.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

cancer = load_breast_cancer(as_frame=True)
df = cancer.frame
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

print("Full classification report:")
print(classification_report(y_test, predictions, target_names=cancer.target_names))

precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"""
Interpretation (for class 1 = "{cancer.target_names[1]}"):
- Precision ({precision:.2f}): of all cases the model predicted as
  "{cancer.target_names[1]}", {precision:.0%} actually were. Low precision
  means lots of false alarms.
- Recall ({recall:.2f}): of all the actual "{cancer.target_names[1]}" cases,
  the model caught {recall:.0%} of them. Low recall means it's missing
  real cases.
- F1-score ({f1:.2f}): the harmonic mean of precision and recall - a single
  number that only looks good when BOTH precision and recall are
  reasonably high, so it won't reward a model that's lopsided (e.g. great
  recall but terrible precision).

Precision and recall usually trade off against each other - a model
tuned to catch more real cases (higher recall) typically flags more
false alarms too (lower precision). Which one matters more depends on
the cost of each error type in your specific use case.
""")