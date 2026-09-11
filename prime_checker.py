"""
Day 4 - Prime Number Checker
A function checks whether a number is prime, tested with different values.
"""


def is_prime(n):
    if n < 2:
        return False  # 0, 1, and negatives are not prime
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def main():
    test_values = [1, 2, 3, 4, 17, 18, 29, 49, 97, -5, 0]

    print("Prime Number Checker")
    print("-" * 25)
    for value in test_values:
        result = "Prime" if is_prime(value) else "Not Prime"
        print(f"{value:>4} -> {result}")


if __name__ == "__main__":
    main()
