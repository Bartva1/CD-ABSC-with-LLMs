### Code to calculate KL divergence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

def kl_divergence(P, Q):
    
    # Compute the KL divergence
    kl = 0
    for idx, x in enumerate(P):
        kl += x * np.log(x/Q[idx])

    return kl

# probability distributions of each dataset
# [positive, neutral, negative]

source_laptop = np.array([1074 / 2360, 486 / 2360, 800 / 2360])
source_rest = np.array([2296 / 3777, 672 / 3777, 809 / 3777])
source_book = np.array([755 / 2803, 1741 / 2803, 307 / 2803])

target_laptop = np.array([254 / 591, 143 / 591, 194 / 591])
target_rest = np.array([596 / 945, 157 / 945, 192 / 945])
target_book = np.array([201 / 701, 423 / 701, 77 / 701])
print(source_book)
print(source_laptop)
print(source_rest)
kl_diverg_dict = {'lapt-rest': kl_divergence(source_rest,source_laptop), 'lapt-book': kl_divergence(source_book,source_laptop),
                  'rest-lapt': kl_divergence(source_laptop,source_rest), 'rest-book': kl_divergence(source_book,source_rest),
                  'book-rest': kl_divergence(source_rest,source_book), 'book-lapt': kl_divergence(source_laptop,source_book)}

acc_results = {'lapt-rest': 0.7502645502645503, 'lapt-book': 0.43081312410841655,
                  'rest-lapt': 0.6666666666666666, 'rest-book': 0.4222539229671897,
                  'book-rest': 0.6497354497354497, 'book-lapt':0.583756345177665}

#acc_results = {'lapt-rest': 0.6167531637241462, 'lapt-book': 0.42530106960587716,
#                  'rest-lapt': 0.598039696537051, 'rest-book': 0.4024576612998107,
#                  'book-rest': 0.5600316185337237, 'book-lapt': 0.5300934103498959}
print(kl_diverg_dict)

kl_series = pd.Series(kl_diverg_dict)
acc_series = pd.Series(acc_results)
print(kl_series)
print(acc_series)

# Compute Pearson correlation
pearson_corr, p1 = pearsonr(kl_series, acc_series)
print(f"Pearson correlation: {pearson_corr}")
print(f"P-value: {p1}")

# Compute Spearman correlation
spearman_corr, p2 = spearmanr(kl_series, acc_series)
print(f"Spearman correlation: {spearman_corr}")
print(f"P-value: {p2}")

# Calculate Kendall's Tau
tau, p_value = kendalltau(kl_series, acc_series)
print(f"Kendall's Tau: {tau}")
print(f"P-value: {p_value}")
