import pandas as pd
from scipy.stats import spearmanr
import numpy as np
from scipy.stats import kendalltau
from scipy.stats import pearsonr


data = pd.read_csv(r'vine.csv')

X = data['pH']
Y = data['alcohol']


corr, pval = spearmanr(X, Y)
print(f"Spearman correlation = {corr:.4f}, p-value = {pval:.4g}")

corr, pval = kendalltau(X, Y)
print(f"Kendall tau = {corr:.4f}, p-value = {pval:.4g}")

Y_log = np.log(Y)

corr, pval = pearsonr(X, Y_log)
print(f"Pearson (X и log(Y)): correlation = {corr:.4f}, p-value = {pval:.4g}")