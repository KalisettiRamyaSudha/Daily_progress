"""
W2D2 - Mean, Median, and Standard Deviation
Calculates mean, median, and standard deviation with NumPy and
compares them against manual (from-scratch) calculations.
"""

import numpy as np


def manual_mean(values):
    return sum(values) / len(values)


def manual_median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return sorted_values[mid]


def manual_std_dev(values):
    mean = manual_mean(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    variance = sum(squared_diffs) / len(values)  # population std dev
    return variance ** 0.5


def main():
    data = [12, 15, 22, 8, 19, 30, 4, 27, 15, 11]
    print("Data:", data)

    # --- NumPy ---
    np_array = np.array(data)
    print("\nNumPy results:")
    print("  Mean:  ", np.mean(np_array))
    print("  Median:", np.median(np_array))
    print("  Std:   ", np.std(np_array))

    # --- Manual ---
    print("\nManual results:")
    print("  Mean:  ", manual_mean(data))
    print("  Median:", manual_median(data))
    print("  Std:   ", manual_std_dev(data))

    # --- Comparison ---
    print("\nMatch check:")
    print("  Mean matches:  ", np.isclose(np.mean(np_array), manual_mean(data)))
    print("  Median matches:", np.isclose(np.median(np_array), manual_median(data)))
    print("  Std matches:   ", np.isclose(np.std(np_array), manual_std_dev(data)))


if __name__ == "__main__":
    main()
