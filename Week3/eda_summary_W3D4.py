"""
W3D4 - EDA Summary
Writes five key observations about a dataset, each supported by a chart.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)
n = 200
departments = rng.choice(["Engineering", "Sales", "Marketing", "HR"], size=n, p=[0.4, 0.3, 0.2, 0.1])
years_experience = np.clip(rng.normal(6, 3.5, size=n), 0, 25).round(1)
base_salary = {"Engineering": 85000, "Sales": 65000, "Marketing": 60000, "HR": 58000}
salary = np.array([base_salary[d] for d in departments]) + years_experience * 1800 + rng.normal(0, 5000, size=n)
outlier_idx = rng.choice(n, size=6, replace=False)
salary[outlier_idx] += rng.uniform(60000, 120000, size=6)
salary = salary.round(0)
satisfaction = np.clip(rng.normal(7, 1.5, size=n), 1, 10).round(1)

df = pd.DataFrame({
    "department": departments,
    "years_experience": years_experience,
    "salary": salary,
    "satisfaction_score": satisfaction,
})

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# --- Observation 1: Department headcount is uneven ---
dept_counts = df["department"].value_counts()
axes[0, 0].bar(dept_counts.index, dept_counts.values, color="#4c72b0")
axes[0, 0].set_title("Obs 1: Headcount by Department")
axes[0, 0].set_ylabel("Count")

# --- Observation 2: Salary is right-skewed (mean > median) ---
axes[0, 1].hist(df["salary"], bins=25, color="#55a868", edgecolor="black")
axes[0, 1].axvline(df["salary"].mean(), color="red", linestyle="--", label="Mean")
axes[0, 1].axvline(df["salary"].median(), color="green", linestyle="-", label="Median")
axes[0, 1].set_title("Obs 2: Salary Is Right-Skewed")
axes[0, 1].legend(fontsize=8)

# --- Observation 3: Salary increases with experience ---
axes[0, 2].scatter(df["years_experience"], df["salary"], alpha=0.5, color="#c44e52")
axes[0, 2].set_title("Obs 3: Salary Rises With Experience")
axes[0, 2].set_xlabel("Years Experience")
axes[0, 2].set_ylabel("Salary")

# --- Observation 4: Engineering pays highest on average ---
avg_salary_by_dept = df.groupby("department")["salary"].mean().sort_values(ascending=False)
axes[1, 0].bar(avg_salary_by_dept.index, avg_salary_by_dept.values, color="#8172b2")
axes[1, 0].set_title("Obs 4: Avg Salary by Department")
axes[1, 0].tick_params(axis="x", rotation=20)

# --- Observation 5: Satisfaction is independent of salary ---
axes[1, 1].scatter(df["salary"], df["satisfaction_score"], alpha=0.5, color="#dd8452")
axes[1, 1].set_title("Obs 5: Satisfaction vs. Salary (weak link)")
axes[1, 1].set_xlabel("Salary")
axes[1, 1].set_ylabel("Satisfaction Score")

axes[1, 2].axis("off")  # unused panel

fig.tight_layout()
fig.savefig("eda_summary_W3D4.png", dpi=150)
print("Saved eda_summary_W3D4.png")

correlation = df["salary"].corr(df["satisfaction_score"])

print("\n=== Five Key Observations ===")
print(f"""
1. Headcount is uneven across departments: Engineering has the most
   employees ({dept_counts['Engineering']}), HR the fewest ({dept_counts['HR']}).
   (See 'Headcount by Department' chart.)

2. Salary is right-skewed: mean (${df['salary'].mean():,.0f}) is noticeably
   higher than median (${df['salary'].median():,.0f}), driven by a handful
   of high earners. (See 'Salary Is Right-Skewed' chart.)

3. Salary rises with years of experience, showing a clear positive trend
   in the scatter plot rather than a flat/random spread.
   (See 'Salary Rises With Experience' chart.)

4. Engineering pays the highest average salary (${avg_salary_by_dept.iloc[0]:,.0f}),
   HR the lowest (${avg_salary_by_dept.iloc[-1]:,.0f}).
   (See 'Avg Salary by Department' chart.)

5. Satisfaction score shows little relationship with salary
   (correlation = {correlation:.2f}) - pay alone doesn't explain how
   satisfied employees report being. (See 'Satisfaction vs. Salary' chart.)
""")