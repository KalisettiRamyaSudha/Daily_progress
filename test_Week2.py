def ascending(values):
    return sorted(values)

def main():
    # Function Usage
    cost = [20, 10, 30, 40, 45, 68, 25, 9, 12]
    sorted_cost = ascending(cost)
    print("Ascending values of the list",sorted_cost)
    # List Indexing print the list in revers
    print("List reversal ", cost[::-1])

    #String Operations
    fruits = ["apple", "banana", "cherry", 123, "chocolate", 56]
    for fruit in fruits:
        print(fruit.isalpha())
if __name__ == "__main__":
    main()
