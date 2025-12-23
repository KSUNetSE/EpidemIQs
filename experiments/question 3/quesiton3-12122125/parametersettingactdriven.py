
import numpy as np

# Given parameters
N = 1000
alpha = 0.1
m = 2
gamma = 0.2  # recovery probability per infected node per time step
R0 = 3.0

# Per-contact infection rate to match R0:
#   R0 = (2*m*alpha * beta) / gamma => beta = R0 * gamma / (2*m*alpha)
beta = R0 * gamma / (2*m*alpha)  # This results in beta=1.5

# But probabilities >1 are not physical in DTMC, so convert to probability:
#   p = 1 - exp(-beta)
p_infect = 1 - np.exp(-beta)

# Initial condition: 1 infected node, N-1 susceptible, 0 removed
num_infected = 1
num_susceptible = N - num_infected
num_removed = 0
ic_percent = [
    {'S': round(100*num_susceptible/N), 'I': round(100*num_infected/N), 'R': 0}
]
# Ensure sum to 100
correction = 100 - (ic_percent[0]['S'] + ic_percent[0]['I'] + ic_percent[0]['R'])
if correction != 0:
    ic_percent[0]['S'] += correction  # correct susceptible to sum to 100

# Save parameters and IC for output
params = {'per_contact_infect_prob': float(np.round(p_infect, 3)), 'gamma': gamma}  # round for readability
return_vars = ['params', 'ic_percent', 'beta', 'p_infect']
