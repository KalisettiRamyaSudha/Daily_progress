"""
Day 4 - Pattern Printing
Three star-pattern programs using nested loops:
1. Right-angled triangle
2. Inverted right-angled triangle
3. Pyramid
"""


def print_right_triangle(rows):
    for i in range(1, rows + 1):
        print("*" * i)


def print_inverted_triangle(rows):
    for i in range(rows, 0, -1):
        print("*" * i)


def print_pyramid(rows):
    for i in range(1, rows + 1):
        spaces = " " * (rows - i)
        stars = "*" * (2 * i - 1)
        print(spaces + stars)


def main():
    rows = 5

    print("Pattern 1: Right-Angled Triangle")
    print_right_triangle(rows)

    print("\nPattern 2: Inverted Right-Angled Triangle")
    print_inverted_triangle(rows)

    print("\nPattern 3: Pyramid")
    print_pyramid(rows)


if __name__ == "__main__":
    main()
