"""
Day 1 - Number Classification Program
Checks whether a number is positive, negative, or zero using if-else conditions.
Tested with multiple sample inputs.
"""


def classify_number(num):
    if num > 0:
        return "Positive"
    elif num < 0:
        return "Negative"
    else:
        return "Zero"


def main():
    # Sample test inputs covering positive, negative, zero, and decimal values
    test_inputs = [10, -5, 0, 3.14, -2.5, 100, -100]

    print("Number Classification Program")
    print("-" * 35)
    for value in test_inputs:
        result = classify_number(value)
        print(f"{value:>8} -> {result}")

    # Let the user try their own values too
    print("\nEnter your own numbers to classify (type 'done' to stop):")
    while True:
        user_input = input("Enter a number: ").strip()
        if user_input.lower() == "done":
            break
        try:
            num = float(user_input)
            print(f"{num} is {classify_number(num)}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


if __name__ == "__main__":
    main()
