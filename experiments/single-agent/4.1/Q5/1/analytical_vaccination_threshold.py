
import numpy as np
from math import factorial, exp

# Problem parameters
R0 = 4
z = 3
k = 10
max_k = 30

# Calculate random vaccination threshold analytically
vc_random = 1 - 1/R0

# Degree distribution (Poisson with mean z)
pk = np.array([exp(-z) * z**k_ / factorial(k_) for k_ in range(max_k + 1)])
pk /= pk.sum()  # normalization for safety

pk10 = pk[k]  # proportion of degree 10

# Remove all nodes with degree 10 and renormalize distribution
pk_wo10 = pk.copy()
pk_wo10[k] = 0
pk_wo10 /= pk_wo10.sum()  # renormalize

z_new = sum(ki*pk_wo10[ki] for ki in range(max_k + 1))
q_new = sum(ki*(ki-1)*pk_wo10[ki] for ki in range(max_k + 1)) / z_new

results = {
    'random_vaccination': vc_random,
    'prop_degree_10_vaccinated': pk10,
    'mean_excess_degree_after_degree_10_removed': q_new,
    'mean_degree_after_degree_10_removed': z_new
}

results