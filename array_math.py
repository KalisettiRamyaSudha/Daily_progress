"""
Day 1 - Array Math Operations
Performs addition, subtraction, multiplication, and division on
NumPy arrays (element-wise).
"""

import numpy as np


def main():
    a = np.array([10, 20, 30, 40])
    b = np.array([2, 4, 5, 8])

    print("Array a:", a)
    print("Array b:", b)

    print("\nAddition (a + b):      ", a + b)
    print("Subtraction (a - b):   ", a - b)
    print("Multiplication (a * b):", a * b)
    print("Division (a / b):      ", a / b)

    # Scalar operations, for comparison
    print("\nScalar addition (a + 5):      ", a + 5)
    print("Scalar multiplication (a * 2):", a * 2)

    # A couple of aggregate operations that build on the above
    print("\nSum of a:", np.sum(a))
    print("Mean of a:", np.mean(a))


if __name__ == "__main__":
    main()
