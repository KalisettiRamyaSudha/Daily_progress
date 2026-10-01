"""
W4D3 - Accuracy Score
Calculates accuracy for classification predictions.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

accuracy = accuracy_score(y_test, predictions)
correct = (predictions == y_test).sum()
total = len(y_test)

print(f"Correct predictions: {correct} / {total}")
print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")

print(f"""
Note: accuracy alone can be misleading on imbalanced datasets. If 95% of
cases were one class, a model that always predicts that class would score
95% accuracy while being useless for the minority class. Here the classes
are reasonably balanced ({y.value_counts().min()} vs {y.value_counts().max()}),
so accuracy is a more trustworthy headline number - but it's still worth
checking the confusion matrix and classification report (next two tasks)
rather than stopping at this one number.
""")