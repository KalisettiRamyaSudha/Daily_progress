"""
Day 3 - Character Frequency
Uses a dictionary to count how many times each character appears
in a string.
"""


def count_characters(text):
    frequency = {}
    for char in text:
        if char in frequency:
            frequency[char] += 1
        else:
            frequency[char] = 1
    return frequency


def main():
    text = input("Enter a string: ")

    frequency = count_characters(text)

    print("\nCharacter frequency:")
    for char, count in frequency.items():
        display_char = char if char != " " else "' '"
        print(f"  {display_char}: {count}")


if __name__ == "__main__":
    main()
