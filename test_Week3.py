import numpy as np
import pandas as pd

matrix_1 = np.array([[2,3,8],[4,5,6]])
matrix_2 = np.array([[4,5,6],[9,7,1]])
matrix_3 = np.array([[7,8,9],[5,0,1]])

"""Matrix operations"""

print("Matrix Sum operation", matrix_1+matrix_2+matrix_3)
print("Matrix product operation", matrix_1*matrix_2)

matrix_4 = np.array([3,4,5,8,9])

print("Mean Values:", np.mean(matrix_4))
matrix_4 = np.array([3,4,5,5,6,7,12,10,24,8,9])
print("Median Values:", np.median(matrix_4))
print("Sorting the array", np.sort(matrix_4))
data = {'name': ["John", "Ray", "May"],
        'salary':[2000, 4000, 5000],
       'dept':["Sales", "IT", "Marketing"]}

df = pd.DataFrame(data)

print("Conversion of dict to dataframe", df)

#Renaming the ID of the dict
df_renamed = df.rename(columns={'name':'Name','salary':'Salary', 'dept':'Department'})
print(df_renamed)

#Groupby department
print("Grouping the dataframe", df.groupby('dept').describe())