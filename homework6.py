import pandas as pd
from scipy.stats import pearsonr
import numpy as np

data = pd.read_csv(r'vine.csv')

X = data['pH']
Y = data['alcohol']
# Если Y логнормальное – логарифмируем его для нормальности
Y_log = np.log(Y)

# Проверка корреляции между X и log(Y)
r, p_value = pearsonr(X, Y_log)

print(f"Коэффициент корреляции Пирсона: r = {r:.4f}")
print(f"p-value: {p_value:.4g}")
