# Daily Progress — Python & Data Analysis

A day-by-day log of exercises building from core Python fundamentals into
NumPy and pandas for data analysis.

## Week 1 - Python Fundamentals
- **Day 1**: Conditionals & control flow — number classification, even/odd checker, menu-driven calculator
- **Day 2**: Lists — append/insert/remove/sort/reverse, manual max/min, sum & average, multiplication tables
- **Day 3**: Strings — length/case/reverse/vowel count, palindrome checker, word counter, character frequency
- **Day 4**: Functions & loops — square/cube/factorial/simple interest, prime checker, Fibonacci series, star patterns
- **Day 5**: File handling & review — safe file read/write/append, student marks & grading, a mini practice set combining Days 1-4, and a Git organization pass

**Key takeaways**: writing reusable single-purpose functions, control flow with validated input, loop-based algorithms (manual max/min, primes, Fibonacci), string manipulation without shortcuts, dictionaries for counting, and safe file I/O with `with`.

## Week 2 - NumPy & pandas
- **Day 1**: NumPy basics — array creation, shape/size/ndim, indexing & slicing, element-wise math
- **Day 2**: Reshape/flatten, random number arrays, mean/median/std dev, matrix operations (element-wise vs. `@` matrix multiplication)
- **Day 3**: pandas Series, DataFrame creation, CSV inspection (`head`/`tail`/`info`/`describe`), column selection & row filtering
- **Day 4**: Missing values (identify/fill/drop), renaming & sorting, calculated columns, `groupby` aggregation
- **Day 5**: A mini dataset project (build → clean → analyze → summarize) and a summary notebook tying together every NumPy/pandas concept from the week

**Key takeaways**: NumPy arrays support fast element-wise math without loops; `reshape()` is usually a view while `flatten()` is always a copy; pandas DataFrames are labeled tables built from dicts, lists, or CSVs; missing data should be handled deliberately (fill vs. drop); `groupby` + aggregation is the core pattern for category-level analysis.
