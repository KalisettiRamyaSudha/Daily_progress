"""
Day 3 - Word Counter
Reads a sentence, counts the number of words, and prints the length
of each word.
"""


def count_words(sentence):
    words = sentence.split()
    return len(words)


def get_word_lengths(sentence):
    words = sentence.split()
    return {word: len(word) for word in words}


def main():
    sentence = input("Enter a sentence: ")

    word_count = count_words(sentence)
    word_lengths = get_word_lengths(sentence)

    print(f"\nNumber of words: {word_count}")
    print("Word lengths:")
    for word, length in word_lengths.items():
        print(f"  {word}: {length}")


if __name__ == "__main__":
    main()
