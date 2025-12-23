
# SIR Parameter Setting for Temporal and Static Network Models in Experiment 2
import math
# Network diagnostics (from construction and above prompt)
N = 10000
mean_a = 0.0141
mean_a2 = 0.00103
m = 1
R0 = 3.0
mean_k = 13.89
mean_k2 = 902.3

# === Temporal Activity-Driven Network ===
def beta_temporal(mu=1.0, use_heterogeneity=False):
    if use_heterogeneity:
        denom = m * (mean_a + math.sqrt(mean_a2))
    else:
        denom = m * mean_a
    return (R0 * mu) / denom

# === Static (Aggregated Random Network) ===
def beta_static(mu=1.0):
    numer = R0 * mu * mean_k
    denom = mean_k2 - mean_k
    return numer / denom

# Select recovery rate (can be changed for illustrative timescale as needed)
mu = 1.0

# Compute infection rates
beta_temp_simple = beta_temporal(mu, use_heterogeneity=False)
beta_temp_hetero = beta_temporal(mu, use_heterogeneity=True)
beta_stat = beta_static(mu)

parameters = {
    'temporal_simple': {'beta': round(beta_temp_simple, 5), 'mu': mu},
    'temporal_hetero': {'beta': round(beta_temp_hetero, 5), 'mu': mu},
    'static': {'beta': round(beta_stat, 5), 'mu': mu},
}

parameters