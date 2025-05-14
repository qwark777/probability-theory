import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
# Список распределений для проверки
distributions = [
    ('Generalized Extreme Value', stats.genextreme),
    ('Weibull', stats.weibull_min),
    ('Log-Logistic', stats.fisk),
    ('Inverse Gamma', stats.invgamma),
    ('Beta', stats.beta),
    ('Gamma', stats.gamma),
    ('Normal', stats.norm),
    ('Exponential', stats.expon)
]
data = pd.read_csv(r'abalone.csv')

data = data['Diameter']
# 3. Проверка всех распределений
results = []
for name, dist in distributions:
    try:
        # Подгонка параметров
        params = dist.fit(data)

        # Тест Колмогорова-Смирнова
        ks_stat, p_value = stats.kstest(data, dist.name, args=params)

        # Вычисление AIC/BIC
        log_likelihood = np.sum(dist.logpdf(data, *params))
        aic = 2 * len(params) - 2 * log_likelihood
        bic = len(params) * np.log(len(data)) - 2 * log_likelihood

        # MSE между гистограммой и PDF
        hist, bins = np.histogram(data, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        pdf_values = dist.pdf(bin_centers, *params)
        mse = mean_squared_error(hist, pdf_values)

        results.append({
            'name': name,
            'params': params,
            'ks_stat': ks_stat,
            'p_value': p_value,
            'aic': aic,
            'bic': bic,
            'mse': mse
        })

    except Exception as e:
        print(f"Ошибка при подборе {name}: {str(e)}")
        continue

# 4. Сортировка результатов по AIC (чем меньше, тем лучше)
results.sort(key=lambda x: x['aic'])

# 5. Вывод результатов
print("\nРезультаты сравнения распределений:")
print("{:<25} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    'Распределение', 'KS-stat', 'p-value', 'AIC', 'BIC', 'MSE'))
for res in results:
    print("{:<25} {:<10.4f} {:<10.4f} {:<10.2f} {:<10.2f} {:<10.4f}".format(
        res['name'], res['ks_stat'], res['p_value'],
        res['aic'], res['bic'], res['mse']))

# 6. Визуализация топ-3 распределений
plt.figure(figsize=(14, 8))
plt.hist(data, bins=50, density=True, alpha=0.3, color='g', label='Данные')

top_distributions = results[:3]
colors = ['r', 'b', 'm']
x = np.linspace(min(data), max(data), 500)

for i, res in enumerate(top_distributions):
    dist = getattr(stats, res['name'].lower().replace('-', ''))
    pdf = dist.pdf(x, *res['params'])
    plt.plot(x, pdf, color=colors[i],
             label=f"{res['name']} (AIC={res['aic']:.1f})")

plt.title('Лучшие распределения')
plt.xlabel('Значения')
plt.ylabel('Плотность')
plt.legend()
plt.show()

# 7. Проверка смеси распределений (дополнительно)
try:
    gmm = GaussianMixture(n_components=2).fit(data.reshape(-1, 1))
    log_likelihood = gmm.score(data.reshape(-1, 1)) * len(data)
    aic_gmm = 2 * gmm._n_parameters() - 2 * log_likelihood
    bic_gmm = gmm._n_parameters() * np.log(len(data)) - 2 * log_likelihood

    print(f"\nGaussian Mixture: AIC={aic_gmm:.2f}, BIC={bic_gmm:.2f}")

    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=50, density=True, alpha=0.3, color='g')
    x = np.linspace(min(data), max(data), 500).reshape(-1, 1)
    plt.plot(x, np.exp(gmm.score_samples(x)), 'r-',
             label=f'GMM (AIC={aic_gmm:.1f})')
    plt.legend()
    plt.title('Смесь гауссовых распределений')
    plt.show()

except Exception as e:
    print(f"\nОшибка при подборе смеси распределений: {str(e)}")