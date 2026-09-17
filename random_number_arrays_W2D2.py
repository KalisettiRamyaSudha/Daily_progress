"""
W2D2 - Random Number Arrays
Generates random integers and floats, then inspects the generated
values and their ranges.
"""

import numpy as np


def main():
    rng = np.random.default_rng(seed=42)  # seeded for reproducible output

    # --- random integers ---
    random_ints = rng.integers(low=1, high=100, size=10)
    print("Random integers (1-99):", random_ints)
    print("Min:", random_ints.min(), "| Max:", random_ints.max())

    # --- random floats ---
    random_floats = rng.random(size=10)  # floats in [0.0, 1.0)
    print("\nRandom floats [0.0, 1.0):", random_floats)
    print("Min:", round(random_floats.min(), 4), "| Max:", round(random_floats.max(), 4))

    # --- random floats in a custom range ---
    low, high = 10.0, 20.0
    scaled_floats = low + random_floats * (high - low)
    print(f"\nScaled floats [{low}, {high}):", np.round(scaled_floats, 2))

    # --- 2D random array ---
    random_2d = rng.integers(low=0, high=10, size=(3, 4))
    print("\nRandom 2D integer array (3x4):\n", random_2d)
    print("Shape:", random_2d.shape)


if __name__ == "__main__":
    main()
