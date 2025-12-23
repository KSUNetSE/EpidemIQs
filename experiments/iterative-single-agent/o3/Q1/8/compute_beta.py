
import numpy as np
stats = {'ER': {'k_mean': 10.0376, 'k2_mean': 110.8}, 'BA': {'k_mean': 9.99, 'k2_mean': 272.5716}}
R0 = 2.5
gamma = 1/4  # recovery rate

betas = {}
for name, s in stats.items():
    q = (s['k2_mean'] - s['k_mean']) / s['k_mean']
    beta = R0 * gamma / q
    betas[name] = beta
print(betas)