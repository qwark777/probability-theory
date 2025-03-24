import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

data = pd.read_csv(r'vine.csv')
column_data = data['pH']


mean = column_data.mean()
std_dev = column_data.std()


# Построение эмпирической функции распределения
n = len(column_data)
x_ecdf = np.sort(column_data)
y_ecdf = np.arange(1, n + 1) / n


alpha = 0.05


n = len(column_data)
epsilon = np.sqrt(np.log(2 / alpha) / (2 * n))

# Нижняя и верхняя границы доверительного интервала
lower_bound = np.maximum(0, y_ecdf - epsilon)
upper_bound = np.minimum(1, y_ecdf + epsilon)

# Построение теоретической функции распределения
x_theoretical = np.linspace(min(column_data), max(column_data), 1000)
y_theoretical = norm.cdf(x_theoretical, mean, std_dev)


plt.figure(figsize=(8, 6))
plt.plot(x_ecdf, y_ecdf, label='Эмпирическая функция распределения', color='blue')
plt.plot(x_theoretical, y_theoretical, label='Теоретическая функция распределения', color='red', linestyle='--')
plt.fill_between(x_ecdf, lower_bound, upper_bound, color='gray', alpha=0.3, label='5% доверительный интервал')
plt.title('Эмпирическая функция распределения с 5% доверительным интервалом')
plt.xlabel('Цена')
plt.ylabel('Вероятность')
plt.legend()
plt.grid(True)
plt.show()


print("Если теоретическая функция распределения лежит в пределах доверительного интервала, распределение данных близко к теоретическому.")