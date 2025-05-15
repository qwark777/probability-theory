import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv(r'vine.csv')
data_cleaned = data.dropna(subset=['pH'])

column_data = data_cleaned['pH']





mean = column_data.mean()
std_dev = column_data.std()

new_sample_size = 1000
new_sample = np.random.normal(mean, std_dev, new_sample_size)

plt.hist(column_data, bins=30, density=True, alpha=0.6, color='g', edgecolor='black', label='Исходная выборка')

plt.hist(new_sample, bins=30, density=True, alpha=0.6, color='b', edgecolor='black', label='Новая выборка')

plt.title('Сравнение гистограмм исходной и новой выборки')
plt.xlabel('Цена')
plt.ylabel('Плотность')
plt.legend()
plt.show()

print("Если гистограммы близки по форме, новая выборка хорошо аппроксимирует исходное распределение.")