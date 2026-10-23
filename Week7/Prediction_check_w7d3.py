"""
Week 7 Day 3 - Prediction Check
Review sample predictions from the trained image classifier and identify
correct and incorrect cases.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras import layers

digits = load_digits()
X_images, y = digits.images, digits.target
X_flat = X_images.reshape(len(X_images), -1) / 16.0

X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X_flat, y, X_images, test_size=0.2, random_state=42, stratify=y
)

model = keras.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.15, verbose=0)

# predict() returns a probability per class; argmax picks the most likely digit
predicted_probs = model.predict(X_test, verbose=0)
predictions = np.argmax(predicted_probs, axis=1)
confidences = predicted_probs.max(axis=1)

correct_mask = predictions == y_test
print(f"Correct: {correct_mask.sum()} / {len(y_test)} ({correct_mask.mean()*100:.1f}%)")
print(f"Incorrect: {(~correct_mask).sum()} / {len(y_test)}")

print("\nSample of 5 CORRECT predictions:")
correct_idx = np.where(correct_mask)[0][:5]
for i in correct_idx:
    print(f"  True: {y_test[i]} | Predicted: {predictions[i]} | Confidence: {confidences[i]:.3f}")

incorrect_idx = np.where(~correct_mask)[0]
print(f"\nAll {len(incorrect_idx)} INCORRECT predictions:")
for i in incorrect_idx:
    print(f"  True: {y_test[i]} | Predicted: {predictions[i]} | Confidence: {confidences[i]:.3f}")

# Visualize a handful of incorrect predictions, if any exist
n_show = min(5, len(incorrect_idx))
if n_show > 0:
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    if n_show == 1:
        axes = [axes]
    for ax, i in zip(axes, incorrect_idx[:n_show]):
        ax.imshow(img_test[i], cmap="gray")
        ax.set_title(f"True: {y_test[i]}\nPred: {predictions[i]} ({confidences[i]:.2f})")
        ax.axis("off")
    plt.suptitle("Misclassified Examples")
    plt.tight_layout()
    plt.savefig("prediction_check_W7D3.png", dpi=150)
    plt.show()
    print(f"\nChart saved: prediction_check_W7D3.png")
else:
    print("\nNo incorrect predictions in this test set - nothing to visualize.")

cm = confusion_matrix(y_test, predictions)
print("\nConfusion matrix (rows=true, cols=predicted):")
print(cm)
most_confused = []
for true_digit in range(10):
    row = cm[true_digit].copy()
    row[true_digit] = 0
    if row.max() > 0:
        confused_with = row.argmax()
        most_confused.append((true_digit, confused_with, row.max()))
if most_confused:
    print("\nMost commonly confused pairs:")
    for true_d, pred_d, count in sorted(most_confused, key=lambda x: -x[2]):
        print(f"  {true_d} mistaken for {pred_d}: {count} time(s)")