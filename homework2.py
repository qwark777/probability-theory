import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta, moment
from scipy.optimize import fsolve

# Исходные данны
data = pd.read_csv(r'vine.csv')

# Пример данных (замените на ваши выборки)
X_list = data['pH']

k_sterdjs = int(1 + np.log2(len(X_list)))

# 1. Построение гистограммы
plt.figure(figsize=(10, 6))
plt.hist(X_list, bins=k_sterdjs, density=True, alpha=0.6, color='g', edgecolor='black')
plt.title('Гистограмма распределения выборки')
plt.xlabel('Значения')
plt.ylabel('Плотность вероятности')
plt.grid(True)
plt.show()
# 2. Предположение о бета-распределении и оценка параметров методом моментов

mu_X = np.mean(X_list)
sigma_X = np.std(X_list)
print(f"Оценка параметров для X: mu = {mu_X:.2f}, sigma = {sigma_X:.2f}")


Y_list = data['alcohol']
plt.figure(figsize=(10, 6))
plt.hist(Y_list, bins=k_sterdjs, density=True, alpha=0.6, color='b', edgecolor='black')
plt.title('Гистограмма распределения выборки')
plt.xlabel('Значения')
plt.ylabel('Плотность вероятности')
plt.grid(True)
plt.show()

log_Y = np.log(Y_list)
mu_log_Y = np.mean(log_Y)
sigma_log_Y = np.std(log_Y)
print(f"Оценка параметров для Y: mu = {mu_log_Y:.2f}, sigma = {sigma_log_Y:.2f}")