
# Set up SEIR model parameters and initial conditions for both networks
# Use realistic SEIR parameters
N = 1000
# Infectious period ~5 days, incubation ~3 days, set beta to get R0 ~2.5 for homogeneous
sigma = 1/3  # incubation rate (E to I)
gamma = 1/5  # recovery rate (I to R)
R0_target = 2.5

# For homogeneous-mixing (ER): beta = R0 * gamma
beta_er = R0_target * gamma
# For BA, beta is computed given R0 = (beta/gamma) * (<k^2>/<k>)
kmean_ba = 7.968
k2_ba = 138.024
beta_ba = R0_target * gamma * kmean_ba / k2_ba

params = {
    'er': {'beta': float(beta_er), 'sigma': sigma, 'gamma': gamma},
    'ba': {'beta': float(beta_ba), 'sigma': sigma, 'gamma': gamma}
}

# Initial conditions: 99% S, 1% E, 0% I, 0% R
ic = {'S': int(0.99*N), 'E': int(0.01*N), 'I': 0, 'R': 0}
params, ic