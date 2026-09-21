"""
Day 2 - Sum and Average Program
Calculates the sum and average of numbers in a list, using functions
to keep the code organized.
"""


def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total


def calculate_average(numbers):
    return calculate_sum(numbers) / len(numbers)


def main():
    numbers = [10, 20, 30, 40, 50]
    print(f"List: {numbers}")
    print(f"Sum: {calculate_sum(numbers)}")
    print(f"Average: {calculate_average(numbers)}")


if __name__ == "__main__":
    main()
