"""
Day 1 - Array Indexing and Slicing
Practices accessing elements, rows, and columns using indexing and
slicing on NumPy arrays.
"""

import numpy as np


def main():
    array_1d = np.array([10, 20, 30, 40, 50])
    print("1D array:", array_1d)
    print("First element:", array_1d[0])
    print("Last element:", array_1d[-1])
    print("Slice [1:4]:", array_1d[1:4])
    print("Every other element [::2]:", array_1d[::2])

    array_2d = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ])
    print("\n2D array:\n", array_2d)

    print("\nElement at row 1, col 2:", array_2d[1, 2])
    print("Entire first row:", array_2d[0])
    print("Entire third column:", array_2d[:, 2])
    print("Sub-matrix (rows 0-1, cols 1-2):\n", array_2d[0:2, 1:3])
    print("Last row:", array_2d[-1])
    print("Last column:", array_2d[:, -1])


if __name__ == "__main__":
    main()
