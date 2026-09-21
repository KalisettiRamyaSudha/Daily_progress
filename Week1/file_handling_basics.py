"""
Day 5 - File Handling Basics
Creates a text file and practices write, append, and read operations
using safe file handling (the "with" statement, which closes the file
automatically, plus try/except for missing-file errors).
"""

FILENAME = "sample.txt"


def write_file(filename, text):
    """Creates the file (or overwrites it) with the given text."""
    with open(filename, "w") as f:
        f.write(text)


def append_file(filename, text):
    """Adds text to the end of an existing file without erasing it."""
    with open(filename, "a") as f:
        f.write(text)


def read_file(filename):
    """Safely reads and returns the file's contents, handling a missing file."""
    try:
        with open(filename, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: '{filename}' does not exist."


def main():
    print("=== Write ===")
    write_file(FILENAME, "Day 5 - File Handling Basics\n")
    print(f"Wrote initial content to {FILENAME}")

    print("\n=== Append ===")
    append_file(FILENAME, "This line was added with append().\n")
    append_file(FILENAME, "Safe file handling uses 'with', so files always close properly.\n")
    print(f"Appended two more lines to {FILENAME}")

    print("\n=== Read ===")
    contents = read_file(FILENAME)
    print(f"Contents of {FILENAME}:")
    print(contents)

    print("=== Read (missing file example) ===")
    print(read_file("does_not_exist.txt"))


if __name__ == "__main__":
    main()
