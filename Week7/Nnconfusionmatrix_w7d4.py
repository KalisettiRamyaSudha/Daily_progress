"""
Week 7 Day 4 - Confusion Matrix for Neural Network
Build and visualize a confusion matrix for the trained neural network's
predictions on the digit classification test set.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow import keras
from keras import layers

digits = load_digits()
X_images, y = digits.images, digits.target
X_flat = X_images.reshape(len(X_images), -1) / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y
)

model = keras.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

cm = confusion_matrix(y_test, predictions)
print("Confusion matrix (rows=true digit, columns=predicted digit):")
print(cm)

fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
ax.set_title("Neural Network Confusion Matrix (Digit Classification)")
plt.tight_layout()
plt.savefig("nn_confusion_matrix_W7D4.png", dpi=150)
plt.show()

print(f"\nChart saved: nn_confusion_matrix_W7D4.png")

print("\nPer-digit classification report:")
print(classification_report(y_test, predictions))

# Find the single most-confused off-diagonal pair
cm_off_diag = cm.copy()
np.fill_diagonal(cm_off_diag, 0)
true_d, pred_d = np.unravel_index(cm_off_diag.argmax(), cm_off_diag.shape)
print(
    f"\nMost common error: true digit {true_d} predicted as {pred_d} "
    f"({cm_off_diag[true_d, pred_d]} times) - everything on the diagonal "
    "is a correct prediction; anything off it is where the model confused "
    "one digit for another."
)