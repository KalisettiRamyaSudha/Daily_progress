"""
Day 3 - Palindrome Checker
Checks whether a given word is a palindrome by reversing the text
manually (without using slicing shortcuts like text[::-1]).
"""


def reverse_manually(text):
    reversed_text = ""
    for char in text:
        reversed_text = char + reversed_text
    return reversed_text


def is_palindrome(text):
    # Ignore case and spaces so phrases like "Racecar" or "nurses run" work too
    cleaned = text.lower().replace(" ", "")
    reversed_text = reverse_manually(cleaned)
    return cleaned == reversed_text


def main():
    word = input("Enter a word or phrase: ")

    if is_palindrome(word):
        print(f'"{word}" is a palindrome.')
    else:
        print(f'"{word}" is not a palindrome.')


if __name__ == "__main__":
    main()
