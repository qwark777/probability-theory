import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'vine.csv')
data_cleaned = data.dropna(subset=['pH'])
column_data = data_cleaned['pH']





n_bins = int(1 + np.log2(len(column_data)))
plt.hist(column_data, bins=n_bins, density=True, alpha=0.6, color='g', edgecolor='black', label='Гистограмма')
hist, bin_edges = np.histogram(column_data, bins=n_bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.plot(bin_centers, hist, 'r-', linewidth=2, label='Полигон частот')


sns.kdeplot(column_data, color='b', linewidth=2, label='Кривая плотности')

plt.title('Полигон частот и кривая плотности распределения')
plt.xlabel('pH')
plt.ylabel('Плотность')
plt.legend()
plt.show()

print("Если кривая плотности близко следует за полигоном частот, распределение данных близко к теоретическому.")
