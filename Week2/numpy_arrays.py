"""
Day 1 - NumPy Array Creation
Creates 1D and 2D arrays and inspects shape, size, and number of dimensions.

Install NumPy first if needed:
    pip install numpy
"""

import numpy as np


def main():
    print("NumPy version:", np.__version__)

    # --- 1D array ---
    array_1d = np.array([10, 20, 30, 40, 50])
    print("\n1D array:", array_1d)
    print("Shape:", array_1d.shape)
    print("Size:", array_1d.size)
    print("Dimensions (ndim):", array_1d.ndim)

    # --- 2D array ---
    array_2d = np.array([[1, 2, 3], [4, 5, 6]])
    print("\n2D array:\n", array_2d)
    print("Shape:", array_2d.shape)
    print("Size:", array_2d.size)
    print("Dimensions (ndim):", array_2d.ndim)

    # --- A few other common creation methods, for reference ---
    print("\nzeros(3, 3):\n", np.zeros((3, 3)))
    print("\nones(2, 4):\n", np.ones((2, 4)))
    print("\narange(0, 10, 2):", np.arange(0, 10, 2))


if __name__ == "__main__":
    main()
