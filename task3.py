import pandas as pd


data = pd.read_csv(r'vine.csv')
column_data = data['pH']


estimated_mean = column_data.mean() # методы моментов реализованы в библиотеке pandas
estimated_variance = column_data.var() # ddof=0 для несмещенной оценки


print(f"Оценка среднего (μ): {estimated_mean}")
print(f"Оценка дисперсии (σ²): {estimated_variance}")
