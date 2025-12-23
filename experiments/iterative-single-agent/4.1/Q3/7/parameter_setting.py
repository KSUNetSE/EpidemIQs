
# SIR parameter setting using discovered mean degree, second moment, and target R0=3
# SIR definitions: beta (infection/transmission rate), gamma (recovery rate)
# For the static case, contact matrix is averaged over time
R0 = 3
k_mean = 181.142
k2_mean = 33003.138

gamma = 1/5  # recovery rate (mean infectious period = 5 time steps)

# beta for static aggregated network: beta = R0 * gamma / ((k2_mean - k_mean) / k_mean)
q = (k2_mean - k_mean) / k_mean
beta_static = R0 * gamma / q

# For temporal network (activity-driven):
# Using the standard approximated epidemic threshold for activity-driven networks: lambda_c = gamma / (m * alpha)
# For R0=3, set beta_temporal = R0 * gamma / (m*alpha)
beta_temporal = R0 * gamma / (m * alpha)


params = {
    'R0': R0,
    'gamma': gamma,
    'beta_static': beta_static,
    'beta_temporal': beta_temporal,
    'q': q,
    'network_mean_degree': k_mean,
    'network_deg2_mean': k2_mean
}

params