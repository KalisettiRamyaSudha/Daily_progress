"""
W2D2 - Matrix Operations
Creates two matrices and performs addition and multiplication
(both element-wise and true matrix multiplication) using NumPy.
"""

import numpy as np


def main():
    matrix_a = np.array([[1, 2, 3],
                          [4, 5, 6]])

    matrix_b = np.array([[7, 8, 9],
                          [10, 11, 12]])

    print("Matrix A:\n", matrix_a)
    print("\nMatrix B:\n", matrix_b)

    # --- Addition (requires same shape) ---
    print("\nA + B (element-wise addition):\n", matrix_a + matrix_b)

    # --- Element-wise multiplication (requires same shape) ---
    print("\nA * B (element-wise multiplication):\n", matrix_a * matrix_b)

    # --- True matrix multiplication (requires A's columns == C's rows) ---
    matrix_c = np.array([[1, 2],
                          [3, 4],
                          [5, 6]])
    print("\nMatrix C (3x2):\n", matrix_c)
    print("\nA @ C (matrix multiplication, 2x3 · 3x2 -> 2x2):\n", matrix_a @ matrix_c)
    print("(Same result via np.matmul):\n", np.matmul(matrix_a, matrix_c))


if __name__ == "__main__":
    main()
