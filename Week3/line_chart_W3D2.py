import matplotlib.pyplot as plt

# Axes

x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 5, 3]

plt.plot(x, y)
plt.title("Simple Line Chart")
plt.xlabel("X Axis Label")
plt.ylabel("Y Axis Label")
plt.savefig('my_line_chart.png', dpi=300, bbox_inches='tight')
print("Chart successfully saved as my_line_chart.png")
