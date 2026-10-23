"""
Week 7 Day 3 - Model Improvement Trial
Adjust epochs, hidden units, and layer depth to try improving on the
basic image classifier's accuracy.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers

warnings.filterwarnings("ignore")

digits = load_digits()
X_images, y = digits.images, digits.target
X_flat = X_images.reshape(len(X_images), -1) / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y
)

def build_and_train(hidden_layers, epochs):
    layer_list = [layers.Input(shape=(64,))]
    for units in hidden_layers:
        layer_list.append(layers.Dense(units, activation="relu"))
    layer_list.append(layers.Dense(10, activation="softmax"))

    model = keras.Sequential(layer_list)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_acc, test_loss

# --- Baseline (from the previous task) ---
baseline_acc, baseline_loss = build_and_train([32, 16], epochs=30)

# --- Trial 1: more epochs, same architecture ---
more_epochs_acc, more_epochs_loss = build_and_train([32, 16], epochs=60)

# --- Trial 2: wider hidden layer ---
wider_acc, wider_loss = build_and_train([64, 32], epochs=30)

# --- Trial 3: deeper network ---
deeper_acc, deeper_loss = build_and_train([64, 32, 16], epochs=30)

# --- Trial 4: wider AND more epochs ---
best_combo_acc, best_combo_loss = build_and_train([64, 32], epochs=60)

results = pd.DataFrame([
    {"trial": "Baseline: (32,16), 30 epochs", "test_accuracy": baseline_acc, "test_loss": baseline_loss},
    {"trial": "More epochs: (32,16), 60 epochs", "test_accuracy": more_epochs_acc, "test_loss": more_epochs_loss},
    {"trial": "Wider: (64,32), 30 epochs", "test_accuracy": wider_acc, "test_loss": wider_loss},
    {"trial": "Deeper: (64,32,16), 30 epochs", "test_accuracy": deeper_acc, "test_loss": deeper_loss},
    {"trial": "Wider + more epochs: (64,32), 60 epochs", "test_accuracy": best_combo_acc, "test_loss": best_combo_loss},
]).round(4)

print(results.to_string(index=False))

best_row = results.loc[results["test_accuracy"].idxmax()]
print(f"\nBest result: {best_row['trial']} -> {best_row['test_accuracy']:.4f} accuracy")
print(
    "\nOn a dataset this small (1,797 images total, 64 input features), "
    "gains from a bigger/deeper network are modest and inconsistent run "
    "to run - the baseline was already capturing most of what's learnable "
    "here. More epochs helps up to a point but risks overfitting a small "
    "dataset further; a wider or deeper network adds capacity that this "
    "amount of data may not need. On real MNIST (70,000 images, 784 "
    "features), these same trials tend to show clearer, more consistent "
    "improvements from added capacity."
)