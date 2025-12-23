
# SIR MODEL PARAMETERS AND INITIAL CONDITIONS
import numpy as np

# Parameters from earlier
mean_k = 8.162
second_moment = 75.05
# Literature: typical recovery rate (gamma) for diseases with ~5 day infectious period
# For example, COVID-19 (Wuhan 2020) gamma ~= 1/5 = 0.2 per day
# Let's choose gamma = 0.2 for a moderately transmissible disease
# R0 is typically 2.5~3 for classic SIR disease, we pick R0=2.5
R0 = 2.5
gamma = 0.2 # Recovery rate
# For SIR on static ER, beta = R0 * gamma / q, q = (<k^2> - <k>)/<k>
k = mean_k
k2 = second_moment
q = (k2 - k) / k
beta = R0 * gamma / q

# Initial conditions: Let's infect 10 individuals, others susceptible in a network of 1000
IC = {'percentage': {'S': 990, 'I': 10, 'R': 0}}  # use integers as required
parameters = {'beta': float(beta), 'gamma': float(gamma)}

(parameters, IC, q, beta, gamma, R0)