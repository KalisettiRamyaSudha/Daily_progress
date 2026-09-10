"""
Day 3 - String Operations Practice
Practices string length, uppercase, lowercase, reverse, and vowel count
operations, each in its own function.
"""


def get_length(text):
    return len(text)


def to_uppercase(text):
    return text.upper()


def to_lowercase(text):
    return text.lower()


def reverse_text(text):
    return text[::-1]


def count_vowels(text):
    vowels = "aeiouAEIOU"
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count


def main():
    text = input("Enter a string: ")

    print(f"\nOriginal string: {text}")
    print(f"Length:          {get_length(text)}")
    print(f"Uppercase:       {to_uppercase(text)}")
    print(f"Lowercase:       {to_lowercase(text)}")
    print(f"Reversed:        {reverse_text(text)}")
    print(f"Vowel count:     {count_vowels(text)}")


if __name__ == "__main__":
    main()
