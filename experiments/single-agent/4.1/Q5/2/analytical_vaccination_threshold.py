
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
import numpy as np

# Analytical vaccination coverage calculations
# Given parameters:
R0 = 4  # Basic reproduction number
q = 4   # Mean excess degree
z = 3   # Mean degree
k = 10  # Degree for degree-targeted vaccination

# For uncorrelated networks, beta and gamma (transmission and recovery rates) can be omitted for coverage calculation
# ---- Scenario 1: Random vaccination ----
# The critical vaccination threshold is generally:  p_c = 1 - 1/R0
pv_rand = 1 - 1/R0

# For more precise network approaches:
# In random uncorrelated networks: p_c_rand = 1 - 1/R0
# (This is true for random vaccination due to random mixing)

# ---- Scenario 2: Degree-targeted vaccination ----
# For nodes of degree k, the equation for final outbreak size and threshold is more complex,
# but the simplest threshold approximates as:
#   p_c_k = (fraction of nodes of degree k to remove to bring effective R0 < 1)
# Because we're only vaccinating degree k=10, must check: what fraction of all nodes have degree 10?
# Let's assume a Poisson degree distribution for the network (sparse, no correlations)
from scipy.stats import poisson

# Probability (fraction) of nodes of degree 10 for mean degree z
p_k10 = poisson.pmf(k, z)

# Total R0 for original network: R0 = transmission/(recovery) * mean excess degree
# By removing only k=10 nodes, what fraction do we need? The critical condition is that the new R0'<1.
# Approximating as:
# Each vaccinated k=10 node removes 10*(10-1) links in denominator of effective R0
# Let's try a simplified approach: After removing all nodes of degree 10, recalculate new <k>, <k^2> and thus new R0_eff
ks = np.arange(0, 25)
Pk = poisson.pmf(ks, z)

# Remove fraction f_k10 of degree 10 nodes
# For a given fraction, recalculate <k> and <k^2>
# Let's search for the minimum fraction (between 0% and 100%) of degree 10 nodes to be vaccinated
fraction_vaccinated = np.linspace(0, 1, 201)
R0_eff = []
new_mean_k = []
new_k2 = []

for frac in fraction_vaccinated:
    new_Pk = Pk.copy()
    new_Pk[10] *= (1 - frac)  # reduce P(degree=10)
    new_Pk = new_Pk/new_Pk.sum()  # re-normalize
    k_vals = ks
    k_mean = np.sum(k_vals * new_Pk)
    k2_mean = np.sum((k_vals**2) * new_Pk)
    q_new = (k2_mean - k_mean) / k_mean if k_mean > 0 else 0
    R0_eff.append(q_new)
    new_mean_k.append(k_mean)
    new_k2.append(k2_mean)

R0_eff = np.array(R0_eff)
# Find the minimum fraction where R0_eff < 1
idx = np.where(R0_eff < 1)[0]
pv_deg = None
if len(idx) > 0:
    pv_deg = fraction_vaccinated[idx[0]]

# Results:
results = dict(pv_rand=pv_rand, pv_deg=pv_deg, p_k10=p_k10)
results