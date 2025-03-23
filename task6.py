import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

data = pd.read_csv(r'automobile_data.csv')
column_data = data['price']

# Нахождение параметров распределения
mean = column_data.mean()
std_dev = column_data.std()

# Построение эмпирической функции распределения
n = len(column_data)
x_ecdf = np.sort(column_data)
y_ecdf = np.arange(1, n + 1) / n

# Построение теоретической функции распределения (CDF)
x_theoretical = np.linspace(min(column_data), max(column_data), 1000)
y_theoretical = norm.cdf(x_theoretical, mean, std_dev)

# Построение графиков
plt.figure(figsize=(8, 6))
plt.plot(x_ecdf, y_ecdf, label='Эмпирическая функция распределения (ECDF)', color='blue')
plt.plot(x_theoretical, y_theoretical, label='Теоретическая функция распределения (CDF)', color='red', linestyle='--')
plt.title('Сравнение эмпирической и теоретической функций распределения')
plt.xlabel('Цена')
plt.ylabel('Кумулятивная вероятность')
plt.legend()
plt.grid(True)
plt.show()

print("Если ECDF близко следует за теоретической CDF, распределение данных близко к теоретическому.")