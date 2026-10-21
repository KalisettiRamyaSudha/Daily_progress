"""
Week 7 Day 2 - Loss and Accuracy Plot
Plot training (and validation) loss and accuracy curves from the Keras
model's training history.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# history.history is a dict: {"loss": [...], "accuracy": [...], "val_loss": [...], "val_accuracy": [...]}
# one value per epoch, which is exactly what's needed to plot training curves
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.15,
    verbose=0,
)

epochs_range = range(1, len(history.history["loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(epochs_range, history.history["loss"], label="Training Loss", marker=".")
axes[0].plot(epochs_range, history.history["val_loss"], label="Validation Loss", marker=".")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (binary crossentropy)")
axes[0].set_title("Loss over Training")
axes[0].legend()

axes[1].plot(epochs_range, history.history["accuracy"], label="Training Accuracy", marker=".")
axes[1].plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", marker=".")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy over Training")
axes[1].legend()

plt.tight_layout()
plt.savefig("loss_accuracy_plot_W7D2.png", dpi=150)
plt.show()

print(f"Chart saved: loss_accuracy_plot_W7D2.png")
print(f"Loss dropped from {history.history['loss'][0]:.4f} to {history.history['loss'][-1]:.4f} over {len(epochs_range)} epochs")
print(f"Accuracy rose from {history.history['accuracy'][0]:.4f} to {history.history['accuracy'][-1]:.4f}")
print(
    "\nWhat to watch for: if training loss keeps dropping while validation "
    "loss flattens out or starts rising, that's the model starting to "
    "overfit - memorizing the training data rather than learning patterns "
    "that generalize. Here both curves move together and flatten out "
    "around the same point, which is a healthy training run."
)