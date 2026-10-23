"""
Week 7 Day 3 - Basic Image Classifier
Build a simple neural network for image classification (10-class digit
recognition, 0-9).
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

# Flatten each 8x8 image into a 64-length vector, and scale pixel values
# from [0, 16] to [0, 1] - neural networks train better on small,
# consistently-scaled numbers.
X_flat = X_images.reshape(len(X_images), -1) / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y
)

# 10-class classification: output layer has 10 neurons (one per digit),
# with 'softmax' turning the raw outputs into a probability distribution
# that sums to 1 across all 10 classes.
model = keras.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # labels are plain integers (0-9), not one-hot
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.15,
    verbose=0,
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal training accuracy:   {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

model.save("image_classifier_W7D3.keras")
print("\nSaved trained model to image_classifier_W7D3.keras")