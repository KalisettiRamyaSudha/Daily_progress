"""
Day 2 - Multiplication Table Generator
Generates a multiplication table from 1 to 10 for a given number using loops.
"""


def print_multiplication_table(number):
    for i in range(1, 11):
        print(f"{number} x {i} = {number * i}")


def main():
    while True:
        try:
            number = int(input("Enter a number to generate its multiplication table: "))
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    print(f"\nMultiplication Table for {number}:")
    print_multiplication_table(number)


if __name__ == "__main__":
    main()
