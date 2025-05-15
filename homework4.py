import numpy as np
import pandas as pd
from scipy.stats import kstest

data = pd.read_csv(r'vine.csv')

X = data['pH']
Y = data['alcohol']
X_std = (X - np.mean(X)) / np.std(X, ddof=0)
Y_log = np.log(Y)
Y_std = (Y_log - np.mean(Y_log)) / np.std(Y_log, ddof=0)

ks_stat_X, ks_pval_X = kstest(X_std, 'norm')
print(f"KS тест для X (нормальное): statistic = {ks_stat_X:.4f}, p-value = {ks_pval_X:.4f}")

ks_stat_Y, ks_pval_Y = kstest(Y_std, 'norm')
print(f"KS тест для Y (логнормальное): statistic = {ks_stat_Y:.4f}, p-value = {ks_pval_Y:.4f}")

from scipy.stats import chisquare, norm, lognorm


def chi_squared_test(data, dist):
    counts, bin_edges = np.histogram(data, bins= int(1 + np.log2(len(data))))
    if dist == 'norm':
        params = norm.fit(data)
        cdf_func = norm.cdf
    elif dist == 'lognorm':
        shape, loc, scale = lognorm.fit(data, floc=0)
        params = (shape, loc, scale)
        cdf_func = lambda x: lognorm.cdf(x, *params)
    else:
        raise ValueError("Unsupported distribution")

    cdf_vals = cdf_func(bin_edges)
    expected_probs = np.diff(cdf_vals)
    expected_counts = expected_probs * len(data)

    expected_counts *= counts.sum() / expected_counts.sum()

    chi2_stat, p_val = chisquare(counts, f_exp=expected_counts)
    return chi2_stat, p_val

chi2_stat_X, p_val_X = chi_squared_test(X, dist='norm')
print(f"χ² для X (нормальное): stat = {chi2_stat_X:.4f}, p = {p_val_X:.4f}")

chi2_stat_Y, p_val_Y = chi_squared_test(Y, dist='lognorm')
print(f"χ² для Y (логнормальное): stat = {chi2_stat_Y:.4f}, p = {p_val_Y:.4f}")