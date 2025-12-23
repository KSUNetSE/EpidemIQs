
import numpy as np

# Parameters
total_nodes = 1000
alpha = 0.1  # activation probability (activity) m = 2  # edges per activation
R0 = 3

# To apply formula for R0 in activity-driven network:
# R0 = (lambda/mu) * m * (E[a] + E[a^2]/E[a]),
# But if all nodes have the same activity alpha, E[a] = alpha, E[a^2] = alpha^2:
# R0 = (lambda/mu) * m * (alpha + alpha^2/alpha) = (lambda/mu) * m * (alpha + alpha) = (lambda/mu) * m * 2*alpha
# => (lambda/mu) * m * 2*alpha = 3, so lambda/mu = 3 / (m * 2 * alpha)
# For SIR on static network: R0 = (beta / gamma) * <k^2>/<k>
# Let's compare beta/gamma for both (to see if recovery/infection rates required differ)

m = 2
alpha = 0.1
E_a = alpha
E_a2 = alpha ** 2

desired_R0 = 3
activity_prefactor = m * (E_a + E_a2/E_a)  # = m * 2*alpha

lambda_over_mu = desired_R0 / activity_prefactor

# Now, let's get mean degree for time-aggregated network for T=100 as example
time_steps = 100
mean_degree_agg = 2 * m * alpha * time_steps  # over T steps (total possible edges in aggregation)
mean_degree_instant = 2 * m * alpha  # network at each time step

# For time-aggregated network, degree distribution is binomial and <k^2> can be approximated (for simplicity, treat as homogeneous for now)
mean_k = mean_degree_instant
second_moment_k = mean_k**2  # since nodes activate independently, approx

# R0 static = (beta/gamma)* <k^2>/<k> = (beta/gamma)*mean_k
# Set to 3 to match R0
beta_over_gamma = desired_R0/mean_k

# Return the derived comparison metrics
return_vars = [
    'lambda_over_mu',
    'activity_prefactor',
    'mean_degree_agg',
    'mean_degree_instant',
    'mean_k',
    'second_moment_k',
    'beta_over_gamma',
]
