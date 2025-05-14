from scipy.stats import chi2
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm
from scipy.optimize import minimize

# Загрузим данные
data = pd.read_csv(r'vine.csv')

# Пример данных (замените на ваши выборки)
X = data['pH']
Y = data['alcohol']

mu_null = 3.22

sigma_mle = np.std(X, ddof=0)
ll_null = np.sum(norm.logpdf(X, loc=mu_null, scale=sigma_mle))

mu_mle = np.mean(X)
ll_alt = np.sum(norm.logpdf(X, loc=mu_mle, scale=sigma_mle))

lrt_stat = -2 * (ll_null - ll_alt)
p_value = 1 - chi2.cdf(lrt_stat, df=1)

print(f"Нормальное μ: LRT statistic: {lrt_stat:.4f}, p-value: {p_value:.4f}")
sigma0 = 0.16
var0 = sigma0 ** 2

mu_mle = np.mean(X)
sigma_mle = np.std(X, ddof=0)

ll_null = np.sum(norm.logpdf(X, loc=mu_mle, scale=sigma0))

ll_alt = np.sum(norm.logpdf(X, loc=mu_mle, scale=sigma_mle))

lrt_stat = -2 * (ll_null - ll_alt)
p_value = 1 - chi2.cdf(lrt_stat, df=1)

print(f"Нормальное σ²: LRT = {lrt_stat:.4f}, p-value = {p_value:.4f}")


log_Y = np.log(Y)

mu_null = 2.34
sigma_mle = np.std(log_Y, ddof=0)

ll_null = np.sum(norm.logpdf(log_Y, loc=mu_null, scale=sigma_mle))
mu_mle = np.mean(log_Y)
ll_alt = np.sum(norm.logpdf(log_Y, loc=mu_mle, scale=sigma_mle))

lrt_stat = -2 * (ll_null - ll_alt)
p_value = 1 - chi2.cdf(lrt_stat, df=1)

print(f"Логнормальное μ: statistic = {lrt_stat:.4f}, p-value = {p_value:.4f}")

log_Y = np.log(Y)

# Гипотеза: σ² логарифма = σ0²
sigma0 = 0.11
mu_mle = np.mean(log_Y)
sigma_mle = np.std(log_Y, ddof=0)

# Log-likelihood при H0
ll_null = np.sum(norm.logpdf(log_Y, loc=mu_mle, scale=sigma0))

# Log-likelihood при H1
ll_alt = np.sum(norm.logpdf(log_Y, loc=mu_mle, scale=sigma_mle))

# LRT
lrt_stat = -2 * (ll_null - ll_alt)
p_value = 1 - chi2.cdf(lrt_stat, df=1)

print(f"Логнормальное σ²: LRT = {lrt_stat:.4f}, p-value = {p_value:.4f}")