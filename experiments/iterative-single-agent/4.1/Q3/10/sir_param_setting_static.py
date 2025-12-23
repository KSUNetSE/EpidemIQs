
# Calculate the SIR infection and recovery rates consistent with R0=3 for the static aggregated network using the degree stats
# SIR: beta = R0 * gamma / q
# For statics: Mean excess degree q = (<k^2> - <k>)/<k>
k_mean = 329.034
k2_mean = 108569.452
R0 = 3
# Let's take an arbitrary but realistic recovery rate, e.g., average infectious period = 5 time steps, so gamma=0.2
gamma = 0.2
q = (k2_mean - k_mean) / k_mean
beta = R0 * gamma / q
(beta, gamma, q)