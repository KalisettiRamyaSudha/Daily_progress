"""
W2D2 - Reshape and Flatten
Uses reshape and flatten on arrays and compares how the output changes.
"""

import numpy as np


def main():
    original = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    print("Original 1D array:", original)
    print("Original shape:", original.shape)

    # --- reshape ---
    reshaped_2d = original.reshape(3, 4)
    print("\nReshaped to (3, 4):\n", reshaped_2d)

    reshaped_2d_alt = original.reshape(4, 3)
    print("\nReshaped to (4, 3):\n", reshaped_2d_alt)

    reshaped_3d = original.reshape(2, 3, 2)
    print("\nReshaped to (2, 3, 2):\n", reshaped_3d)

    # -1 lets NumPy infer that dimension automatically
    reshaped_auto = original.reshape(3, -1)
    print("\nReshaped to (3, -1) [NumPy infers 4 columns]:\n", reshaped_auto)

    # --- flatten ---
    flattened = reshaped_2d.flatten()
    print("\nFlattened back to 1D:", flattened)
    print("Flattened shape:", flattened.shape)

    # Key difference: flatten() always returns a copy, reshape() usually
    # returns a view into the same underlying data (when possible).
    flattened[0] = 999
    print("\nAfter modifying flattened[0], original reshaped_2d is unchanged:")
    print(reshaped_2d)

    reshaped_2d[0, 0] = 999
    print("\nBut modifying reshaped_2d[0,0] DOES affect original (it's a view):")
    print("original:", original)


if __name__ == "__main__":
    main()
