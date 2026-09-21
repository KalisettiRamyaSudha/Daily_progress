"""
Day 2 - Find Largest and Smallest
Finds the maximum and minimum values in a list manually using loops,
instead of the built-in max()/min() functions.
"""


def find_max(numbers):
    largest = numbers[0]
    for num in numbers:
        if num > largest:
            largest = num
    return largest


def find_min(numbers):
    smallest = numbers[0]
    for num in numbers:
        if num < smallest:
            smallest = num
    return smallest


def main():
    numbers = [23, 5, 78, 12, 45, 2, 90, 34]
    print(f"List: {numbers}")
    print(f"Maximum: {find_max(numbers)}")
    print(f"Minimum: {find_min(numbers)}")


if __name__ == "__main__":
    main()
