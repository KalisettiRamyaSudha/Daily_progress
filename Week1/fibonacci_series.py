"""
Day 4 - Fibonacci Series
Generates the Fibonacci series for N terms using loops.
Each Fibonacci number is the sum of the two numbers before it,
starting with 0 and 1.
"""


def generate_fibonacci(n):
    series = []

    # The series starts with 0 and 1 by definition
    if n >= 1:
        series.append(0)
    if n >= 2:
        series.append(1)

    # From the 3rd term onward, each term = sum of the previous two terms
    for i in range(2, n):
        next_term = series[i - 1] + series[i - 2]
        series.append(next_term)

    return series


def main():
    while True:
        try:
            n = int(input("Enter the number of terms: "))
            if n < 0:
                print("Please enter a non-negative number.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    series = generate_fibonacci(n)
    print(f"\nFibonacci series ({n} terms):")
    print(series)


if __name__ == "__main__":
    main()
