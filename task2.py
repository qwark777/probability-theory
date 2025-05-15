import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(r'vine.csv')


column_data = data['pH']


# это коэф стреджиса
k_sterdjs = int(1 + np.log2(len(column_data)))

print(len(column_data), k_sterdjs)
# печать гистограммы

plt.hist(column_data, bins=k_sterdjs, edgecolor='black')
plt.title('Гистограмма выборки')
plt.xlabel('pH')
plt.ylabel('Частота')
plt.show()