"""
Week 7 Day 4 - Error Analysis
Inspect the neural network's wrong predictions in detail and suggest
possible reasons for each type of error.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
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
model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

predicted_probs = model.predict(X_test, verbose=0)
predictions = np.argmax(predicted_probs, axis=1)
confidences = predicted_probs.max(axis=1)

wrong_mask = predictions != y_test
wrong_idx = np.where(wrong_mask)[0]
print(f"Total errors: {len(wrong_idx)} / {len(y_test)} ({len(wrong_idx)/len(y_test)*100:.1f}%)")

# Show, for each error, the top-2 predicted classes (not just the winner) -
# a close second place is a different kind of mistake than a confident
# wrong answer
print("\nAll errors, with top-2 predicted classes and confidence:")
for i in wrong_idx:
    probs = predicted_probs[i]
    top2_idx = np.argsort(probs)[-2:][::-1]
    print(
        f"  True: {y_test[i]} | Predicted: {predictions[i]} ({probs[predictions[i]]:.3f}) | "
        f"Runner-up: {top2_idx[1]} ({probs[top2_idx[1]]:.3f}) | "
        f"{'LOW confidence (<0.6)' if confidences[i] < 0.6 else 'confident but wrong'}"
    )

low_conf_errors = (confidences[wrong_idx] < 0.6).sum()
confident_errors = len(wrong_idx) - low_conf_errors
print(f"\nLow-confidence errors (model was unsure, <0.6): {low_conf_errors}")
print(f"Confident-but-wrong errors (model was sure, >=0.6): {confident_errors}")

n_show = min(6, len(wrong_idx))
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for ax, i in zip(axes.flat, wrong_idx[:n_show]):
    ax.imshow(img_test[i], cmap="gray")
    ax.set_title(f"True: {y_test[i]} | Pred: {predictions[i]} ({confidences[i]:.2f})")
    ax.axis("off")
for ax in axes.flat[n_show:]:
    ax.axis("off")
plt.suptitle("Error Analysis: Misclassified Digits")
plt.tight_layout()
plt.savefig("error_analysis_W7D4.png", dpi=150)
plt.show()
print(f"\nChart saved: error_analysis_W7D4.png")

print(
    "\nPossible reasons for these errors:\n"
    "1. Low resolution (8x8 pixels): at this size, digits that share "
    "structural features - like 8 and 1 (both have a tall, narrow vertical "
    "stroke when handwriting is slanted or the loops of the 8 are drawn "
    "thin/closed) - lose the detail that would disambiguate them at higher "
    "resolution (MNIST's 28x28 would likely reduce this specific confusion).\n"
    "2. Messy/ambiguous handwriting: some digits in any handwritten "
    "dataset are genuinely written ambiguously - a 9 with an open loop can "
    "look like a 7, and a 6 with a short tail can look like a 1.\n"
    "3. Limited training data per class (~145 images/digit in the training "
    "split): less data means less exposure to each digit's variations in "
    "writing style, which makes it easier for an unusual-looking real "
    "example to get mapped to the wrong, more 'typical' class.\n"
    "4. Simple architecture: a small 2-hidden-layer Dense network (no "
    "convolutional layers) has no way to recognize shifted/rotated local "
    "patterns the way a CNN would - a real MNIST CNN typically pushes "
    "error rates below 1%, well under this model's ~3%."
)