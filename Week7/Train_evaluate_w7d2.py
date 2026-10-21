"""
Week 7 Day 2 - Train and Evaluate Model
Train the Dense model from the previous task, evaluate it on the test
set, and generate predictions. Self-contained (rebuilds the same
architecture) so it can be run on its own.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
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

# Train for 50 epochs, holding out 15% of the training data as a
# validation split so we can watch for overfitting during training.
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.15,
    verbose=0,
)

print(f"Training finished after {len(history.history['loss'])} epochs")
print(f"Final training accuracy:   {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Evaluate on the held-out test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Generate predictions: model outputs probabilities (sigmoid), so round to
# get class predictions
predicted_probs = model.predict(X_test_scaled, verbose=0).flatten()
predictions = (predicted_probs > 0.5).astype(int)

print(f"\nFirst 10 predicted probabilities: {predicted_probs[:10].round(3)}")
print(f"First 10 predicted classes:       {predictions[:10]}")
print(f"First 10 actual classes:          {y_test.values[:10]}")

sk_accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy (recomputed with sklearn, should match Keras): {sk_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=cancer.target_names))