"""
Day 5 - Mini Practice Set
Three short problems that combine concepts from Days 1-4.
"""


# ---------------------------------------------------------------------------
# Problem 1: Number classification + even/odd (Day 1)
# For each number in a list, report positive/negative/zero AND even/odd.
# ---------------------------------------------------------------------------
def classify_sign(num):
    if num > 0:
        return "Positive"
    elif num < 0:
        return "Negative"
    else:
        return "Zero"


def classify_parity(num):
    return "Even" if num % 2 == 0 else "Odd"


def problem_1():
    numbers = [15, -8, 0, 7, -4, 22]
    print("Problem 1: Sign + Parity Classification")
    for num in numbers:
        print(f"  {num:>4} -> {classify_sign(num)}, {classify_parity(num)}")


# ---------------------------------------------------------------------------
# Problem 2: Longest word + manual reverse + palindrome check (Day 3)
# Find the longest word in a sentence, reverse it manually, and check
# whether that word is a palindrome.
# ---------------------------------------------------------------------------
def find_longest_word(sentence):
    words = sentence.split()
    longest = words[0]
    for word in words:
        if len(word) > len(longest):
            longest = word
    return longest


def reverse_manually(text):
    reversed_text = ""
    for char in text:
        reversed_text = char + reversed_text
    return reversed_text


def problem_2():
    sentence = "the level of noon deadline reset never civic radar"
    longest = find_longest_word(sentence)
    reversed_word = reverse_manually(longest)
    is_palindrome = longest.lower() == reversed_word.lower()

    print("\nProblem 2: Longest Word + Palindrome Check")
    print(f"  Sentence: {sentence}")
    print(f"  Longest word: {longest}")
    print(f"  Reversed manually: {reversed_word}")
    print(f"  Is palindrome: {is_palindrome}")


# ---------------------------------------------------------------------------
# Problem 3: Fibonacci series + prime count (Day 4)
# Generate the Fibonacci series for N terms, then count how many of
# those terms are prime numbers.
# ---------------------------------------------------------------------------
def generate_fibonacci(n):
    series = []
    if n >= 1:
        series.append(0)
    if n >= 2:
        series.append(1)
    for i in range(2, n):
        series.append(series[i - 1] + series[i - 2])
    return series


def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def problem_3():
    n = 10
    series = generate_fibonacci(n)
    prime_terms = [num for num in series if is_prime(num)]

    print("\nProblem 3: Fibonacci Series + Prime Count")
    print(f"  Fibonacci ({n} terms): {series}")
    print(f"  Prime terms: {prime_terms}")
    print(f"  Count of primes: {len(prime_terms)}")


def main():
    problem_1()
    problem_2()
    problem_3()


if __name__ == "__main__":
    main()
