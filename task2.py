import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(r'automobile_data.csv')


column_data = data['price']

# это коэф стреджиса
k_sterdjs = int(1 + np.log2(len(column_data)))


# печать гистограммы

plt.hist(column_data, bins=k_sterdjs, edgecolor='black')
plt.title('Гистограмма выборки')
plt.xlabel('Цена')
plt.ylabel('Частота')
plt.show()