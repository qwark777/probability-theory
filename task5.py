import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv(r'automobile_data.csv')
data_cleaned = data.dropna(subset=['price'])

# Извлечение столбца для анализа
column_data = data_cleaned['price']




mean = column_data.mean()
std_dev = column_data.std()

# Генерация новой выборки методом обратных функций
new_sample_size = 1000  # Размер новой выборки
new_sample = np.random.normal(mean, std_dev, new_sample_size)

# Построение гистограммы исходной выборки
plt.hist(column_data, bins=30, density=True, alpha=0.6, color='g', edgecolor='black', label='Исходная выборка')

# Построение гистограммы новой выборки
plt.hist(new_sample, bins=30, density=True, alpha=0.6, color='b', edgecolor='black', label='Новая выборка')

# Настройка графика
plt.title('Сравнение гистограмм исходной и новой выборки')
plt.xlabel('Цена')
plt.ylabel('Плотность')
plt.legend()
plt.show()

# Выводы
print("Если гистограммы близки по форме, новая выборка хорошо аппроксимирует исходное распределение.")