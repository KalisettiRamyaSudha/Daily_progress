"""
Week 7 Day 2 - TensorFlow or Keras Setup
Verify the TensorFlow/Keras install and walk through the structure of a
simple Sequential model.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # quiet TF's own startup logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # suppress the oneDNN info banner

import tensorflow as tf
from tensorflow import keras
from keras import layers

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# A Sequential model is the simplest Keras model type: a plain stack of
# layers, each feeding directly into the next, with exactly one input and
# one output.
model = keras.Sequential([
    layers.Input(shape=(10,)),           # 10 input features
    layers.Dense(16, activation="relu"), # hidden layer: 16 neurons
    layers.Dense(8, activation="relu"),  # hidden layer: 8 neurons
    layers.Dense(1, activation="sigmoid"), # output layer: 1 neuron (binary classification)
])

print("\nModel structure:")
model.summary()

total_params = model.count_params()
print(f"\nTotal trainable parameters: {total_params:,}")
print(
    "\nEach Dense layer is 'fully connected' - every neuron in it connects "
    "to every neuron in the previous layer, each connection with its own "
    "weight, plus one bias per neuron. 'relu' lets hidden layers learn "
    "non-linear patterns; 'sigmoid' on the output squashes the final "
    "value into a 0-1 range, read as a probability for binary classification."
)