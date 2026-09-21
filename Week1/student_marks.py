"""
Day 5 - Student Marks Program
Stores marks for five students (three subjects each) in a dictionary
and calculates the average and grade for each student.
"""

students = {
    "Asha": [85, 90, 78],
    "Ravi": [60, 55, 70],
    "Priya": [95, 92, 98],
    "Kiran": [40, 35, 50],
    "Meera": [72, 68, 75],
}


def calculate_average(marks):
    return sum(marks) / len(marks)


def calculate_grade(average):
    if average >= 90:
        return "A"
    elif average >= 75:
        return "B"
    elif average >= 60:
        return "C"
    elif average >= 40:
        return "D"
    else:
        return "F"


def main():
    print(f"{'Student':<10}{'Marks':<20}{'Average':<10}{'Grade':<5}")
    print("-" * 45)
    for name, marks in students.items():
        average = calculate_average(marks)
        grade = calculate_grade(average)
        print(f"{name:<10}{str(marks):<20}{average:<10.2f}{grade:<5}")


if __name__ == "__main__":
    main()
