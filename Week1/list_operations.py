"""
Day 2 - List Operations Practice
Creates a list and practices append, insert, remove, sort, and reverse
operations, printing the result after each operation.
"""


def main():
    numbers = [5, 2, 9, 1, 7]
    print(f"Initial list:        {numbers}")

    numbers.append(12)
    print(f"After append(12):     {numbers}")

    numbers.insert(2, 100)
    print(f"After insert(2, 100): {numbers}")

    numbers.remove(9)
    print(f"After remove(9):      {numbers}")

    numbers.sort()
    print(f"After sort():         {numbers}")

    numbers.reverse()
    print(f"After reverse():      {numbers}")


if __name__ == "__main__":
    main()
