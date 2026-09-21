"""
Day 1 - Even/Odd Checker
Takes five numbers as input and uses a loop to determine whether each
number is even or odd. 
"""


def check_even_odd(num):
    if num % 2 == 0:
        return "Even"
    else:
        return "Odd"


def main():
    numbers = []
    print("Enter 5 numbers:")
    for i in range(5):
        while True:
            try:
                num = int(input(f"Number {i + 1}: "))
                numbers.append(num)
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")

    print("\nResults:")
    for num in numbers:
        print(f"{num} is {check_even_odd(num)}")


if __name__ == "__main__":
    main()
