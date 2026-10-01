"""
W4D3 - Confusion Matrix
Generates a confusion matrix and explains True Positives, True
Negatives, False Positives, and False Negatives.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# labels=[0, 1] makes the row/column order explicit:
# 0 = malignant, 1 = benign (per cancer.target_names)
cm = confusion_matrix(y_test, predictions, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("Confusion matrix (rows=actual, columns=predicted):")
print(cm)
print(f"""
Reading it as TP/TN/FP/FN (treating class 1 = "{cancer.target_names[1]}" as positive):
  True Positives  (TP={tp}): predicted "{cancer.target_names[1]}", actually "{cancer.target_names[1]}"
  True Negatives  (TN={tn}): predicted "{cancer.target_names[0]}", actually "{cancer.target_names[0]}"
  False Positives (FP={fp}): predicted "{cancer.target_names[1]}", actually "{cancer.target_names[0]}" (a false alarm)
  False Negatives (FN={fn}): predicted "{cancer.target_names[0]}", actually "{cancer.target_names[1]}" (a missed case)

In a medical context like this one, False Negatives are usually the most
costly error - a malignant case predicted as benign could mean a missed
diagnosis, which matters more than a False Positive that just leads to
further (unnecessary) testing.
""")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cancer.target_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Breast Cancer Classification")
plt.tight_layout()
plt.savefig("confusion_matrix_W4D3.png", dpi=150)
print("Saved confusion_matrix_W4D3.png")