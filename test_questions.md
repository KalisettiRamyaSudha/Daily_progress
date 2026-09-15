# Week 2, Day 1 — Coding Test: Python Logic and Functions

**Time limit:** 50 minutes total (roughly 10 minutes per question)
**Topics covered:** loops, strings, lists, functions
**Instructions:** Solve each problem as a standalone function. You may use
built-in functions unless a question says otherwise. Aim for working code
first, then clean it up if time remains.

---

### Question 1 — Sum of Even Numbers (loops)
Write a function `sum_of_evens(n)` that returns the sum of all even numbers
from 1 to `n` (inclusive), using a loop.

```python
sum_of_evens(10)  # -> 30  (2+4+6+8+10)
```

---

### Question 2 — Anagram Checker (strings)
Write a function `is_anagram(word1, word2)` that returns `True` if the two
words are anagrams of each other (same letters, same counts, order doesn't
matter), and `False` otherwise. Ignore case.

```python
is_anagram("listen", "silent")  # -> True
is_anagram("hello", "world")    # -> False
```

---

### Question 3 — Remove Duplicates (lists)
Write a function `remove_duplicates(items)` that returns a new list with
duplicate values removed, preserving the original order of first occurrence.
Do not use `set()` for this one — use a loop instead.

```python
remove_duplicates([1, 2, 2, 3, 1, 4])  # -> [1, 2, 3, 4]
```

---

### Question 4 — Second Largest (loops)
Write a function `second_largest(numbers)` that returns the second-largest
distinct value in a list, without using `sorted()` or `max()`.

```python
second_largest([10, 5, 8, 20, 20, 3])  # -> 10
```

---

### Question 5 — Word Length Map (functions + lists + strings, combined)
Write a function `word_lengths(words)` that takes a list of words and
returns a dictionary mapping each word to its length. Then write a second
function `sorted_by_length(word_map)` that returns a list of words sorted
from shortest to longest.

```python
lengths = word_lengths(["banana", "kiwi", "fig", "apple"])
# -> {"banana": 6, "kiwi": 4, "fig": 3, "apple": 5}

sorted_by_length(lengths)
# -> ["fig", "kiwi", "apple", "banana"]
```

---

## Submission
Save your answers as a single Python file (`answers.py`) with one function
per question. Reference solutions are provided separately in `solutions.py`
for self-grading — don't peek until time is up!
