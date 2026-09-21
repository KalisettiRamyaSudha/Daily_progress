"""
Week 2, Day 1 - Coding Test: Reference Solutions
Covers loops, strings, lists, and functions.
See test_questions.md for the problem statements.
"""


# ---------------------------------------------------------------------------
# Q1: Sum of Even Numbers
# ---------------------------------------------------------------------------
def sum_of_evens(n):
    total = 0
    for num in range(1, n + 1):
        if num % 2 == 0:
            total += num
    return total


# ---------------------------------------------------------------------------
# Q2: Anagram Checker
# ---------------------------------------------------------------------------
def is_anagram(word1, word2):
    word1 = word1.lower().replace(" ", "")
    word2 = word2.lower().replace(" ", "")

    if len(word1) != len(word2):
        return False

    return sorted(word1) == sorted(word2)


# ---------------------------------------------------------------------------
# Q3: Remove Duplicates
# ---------------------------------------------------------------------------
def remove_duplicates(items):
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


# ---------------------------------------------------------------------------
# Q4: Second Largest
# ---------------------------------------------------------------------------
def second_largest(numbers):
    largest = second = None
    for num in numbers:
        if largest is None or num > largest:
            if largest is not None and num != largest:
                second = largest
            largest = num
        elif (second is None or num > second) and num != largest:
            second = num
    return second


# ---------------------------------------------------------------------------
# Q5: Word Length Map
# ---------------------------------------------------------------------------
def word_lengths(words):
    return {word: len(word) for word in words}


def sorted_by_length(word_map):
    return sorted(word_map, key=lambda word: word_map[word])


def main():
    print("Q1:", sum_of_evens(10))
    print("Q2:", is_anagram("listen", "silent"), "/", is_anagram("hello", "world"))
    print("Q3:", remove_duplicates([1, 2, 2, 3, 1, 4]))
    print("Q4:", second_largest([10, 5, 8, 20, 20, 3]))

    lengths = word_lengths(["banana", "kiwi", "fig", "apple"])
    print("Q5:", lengths, "->", sorted_by_length(lengths))


if __name__ == "__main__":
    main()
