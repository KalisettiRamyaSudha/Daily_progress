"""
Week 7 Day 2 - Simple Dense Model
Build a dense (fully-connected) neural network in Keras for a basic
binary classification task (Breast Cancer, for continuity with earlier
weeks).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

# Neural networks are sensitive to feature scale, same as MLPClassifier
# yesterday - scale before feeding data in.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()
print(f"\nInput features: {X_train_scaled.shape[1]}")
print(f"Training samples: {len(X_train_scaled)} | Test samples: {len(X_test_scaled)}")
print(
    "\nModel is built and compiled but not yet trained - 'compile' just "
    "wires up the optimizer (how weights get updated), the loss function "
    "(what 'wrong' means for this problem), and the metrics to track. "
    "Training itself is the next task."
)

# Save for the next script to load and continue from
model.save("dense_model_W7D2.keras")
print("\nSaved untrained model to dense_model_W7D2.keras")