"""
Week 7 Day 4 - Save and Load Model
Train a model, save it to disk, then reload it in a fresh session and
use it for prediction - confirming the reloaded model behaves identically
to the one that was trained.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers

digits = load_digits()
X_images, y = digits.images, digits.target
X_flat = X_images.reshape(len(X_images), -1) / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train and save ---
model = keras.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

original_preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
original_accuracy = (original_preds == y_test).mean()
print(f"Original model test accuracy: {original_accuracy:.4f}")

model.save("digit_classifier_W7D4.keras")
print("Saved model to digit_classifier_W7D4.keras")

# --- Delete the in-memory model, to prove the reload is doing real work ---
del model

# --- Reload from disk ---
reloaded_model = keras.models.load_model("digit_classifier_W7D4.keras")
print("\nReloaded model from disk.")

reloaded_preds = np.argmax(reloaded_model.predict(X_test, verbose=0), axis=1)
reloaded_accuracy = (reloaded_preds == y_test).mean()
print(f"Reloaded model test accuracy: {reloaded_accuracy:.4f}")

predictions_match = np.array_equal(original_preds, reloaded_preds)
print(f"\nPredictions identical to the original model: {predictions_match}")

print(
    "\nmodel.save() writes the full model - architecture, trained weights, "
    "optimizer state - to a single .keras file. load_model() rebuilds an "
    "identical model from that file, so it can make predictions "
    "immediately without retraining - this is what lets a trained model "
    "be shipped and used elsewhere (a web app, another script) without "
    "needing access to the original training code or data."
)