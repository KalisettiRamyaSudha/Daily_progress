"""
Day 4 - Functions Practice
Four functions: square, cube, factorial, and simple interest.
Each is called with sample inputs.
"""


def square(n):
    return n ** 2


def cube(n):
    return n ** 3


def factorial(n):
    if n < 0:
        return None  # factorial is undefined for negative numbers
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def simple_interest(principal, rate, time):
    return (principal * rate * time) / 100


def main():
    print("=== Square ===")
    for num in [2, 5, 9]:
        print(f"square({num}) = {square(num)}")

    print("\n=== Cube ===")
    for num in [2, 3, 4]:
        print(f"cube({num}) = {cube(num)}")

    print("\n=== Factorial ===")
    for num in [0, 1, 5, 7]:
        print(f"factorial({num}) = {factorial(num)}")

    print("\n=== Simple Interest ===")
    # simple_interest(principal, rate %, time in years)
    sample_cases = [(1000, 5, 2), (5000, 8, 3), (2500, 10, 1)]
    for principal, rate, time in sample_cases:
        interest = simple_interest(principal, rate, time)
        print(f"simple_interest(P={principal}, R={rate}%, T={time}yr) = {interest}")


if __name__ == "__main__":
    main()
