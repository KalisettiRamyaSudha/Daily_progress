"""
Week 7 Day 3 - Image Data Introduction
Load a handwritten-digit image dataset and inspect its features and
labels before building any model.

NOTE: This sandboxed environment's network access doesn't reach the
hosts that serve the full MNIST dataset (storage.googleapis.com /
openml.org), so this uses scikit-learn's built-in `load_digits` dataset
instead - the same idea (handwritten digits 0-9 as pixel-grid images),
just smaller (8x8 pixels, 1797 images total, vs. MNIST's 28x28, 70,000
images) and bundled directly with scikit-learn, so no download is needed.
Everything below - flattening, Dense layers, predictions - works
identically on real MNIST if you run this with network access.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X_images, y = digits.images, digits.target

print(f"Dataset: {X_images.shape[0]} images, each {X_images.shape[1]}x{X_images.shape[2]} pixels")
print(f"Pixel value range: {X_images.min()} to {X_images.max()} (grayscale intensity)")
print(f"Label dtype: {y.dtype}, unique labels: {sorted(set(y.tolist()))}")

print("\nClass balance:")
unique, counts = np.unique(y, return_counts=True)
for digit, count in zip(unique, counts):
    print(f"  digit {digit}: {count} examples")

print(f"\nA single image as raw data (digit={y[0]}):")
print(X_images[0])

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_images[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")
plt.suptitle("First 10 Digit Images")
plt.tight_layout()
plt.savefig("image_data_intro_W7D3.png", dpi=150)
plt.show()

print(
    "\nUnlike every dataset used so far (rows of named feature columns), "
    "an image is a grid of pixel intensities - there's no 'age' or "
    "'income' column, just raw brightness values arranged spatially. A "
    "simple Dense-layer model needs this grid flattened into one long "
    f"vector first ({X_images.shape[1]}x{X_images.shape[2]} = "
    f"{X_images.shape[1]*X_images.shape[2]} values per image) before it "
    "can be fed in - that flattening happens in the next task."
)