
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
# SIR model: beta (infection rate) and gamma (recovery rate)
# We will set R0=2.5 as typical for COVID-19. 
# gamma (recovery rate) is commonly set as 1/7 (~0.14), for infection duration of ~7 days.
# For ER graph, the mean excess degree, q = (<k^2> - <k>) / <k>
q = (107.852-9.906)/9.906
R0 = 2.5
gamma = 0.142857 # ~1/7
beta = R0*gamma/q

# Set initial conditions: 990 S, 10 I, 0 R (1% initial prevalence)
init_conditions = {'S': 990, 'I': 10, 'R': 0}

with open(os.path.join("output", "sir_params.txt"), "w") as f:
    f.write(f"beta: {beta}\ngamma: {gamma}\nmean_k: {9.906}\nk2: {107.852}\nR0: {R0}\nIC: {init_conditions}\n")

beta, gamma, init_conditions, q
# Analytical calculation of beta for SIR on this network
# Provided parameters from literature review and online search
R0 = 2.8  # Reasonable for general epidemic (COVID-like) as motivation

gamma = 0.07  # Recovery rate, corresponding to avg 14 days infectious period (7%/day recovery)

# On a static network, mean excess degree q = (<k^2>-<k>)/<k>
mean_k = 7.968
mean_k2 = 138.024
q = (mean_k2 - mean_k)/mean_k

# For SIR on network: \beta = R0 * gamma / q
beta = R0 * gamma / q

# Format output for report
rates = {'R0': R0, 'gamma_per_day': gamma, 'mean_k': mean_k, 'mean_k2': mean_k2, 'excess_degree_q': q, 'beta_per_contact_per_day': beta}
rates
# Parameter estimation for SIR model on ER network
# Using literature: R0 = beta * <k^2 - <k>> / (<k> * gamma) for SIR on static network
# We set R0 to 1.8 as per literature. mean_k and second_moment_k from generated network.
mean_k = 5.96
second_moment_k = 41.7333
R0 = 1.8

gamma = 0.1  # Recovery rate, typical for SIR (1/duration of infection in time units)
q = (second_moment_k - mean_k) / mean_k
beta = R0 * gamma / q
params = {'beta': beta, 'gamma': gamma, 'R0': R0, 'mean_k': mean_k, 'second_moment_k': second_moment_k}

# Initial conditions, assume 1% infected at t=0, rest susceptible, 0 recovered.
N = 300
n_I = max(1, int(N * 0.01))
n_S = N - n_I
n_R = 0
init_cond = { 'S': n_S, 'I': n_I, 'R': n_R }

# Plan for parameterization & initial condition setup:
# 1. Use SIR with beta and gamma for Barabasi-Albert network using mean degree and 2nd moment.
# 2. Use literature: typical R0 for COVID/respiratory: 2.6; recovery rate gamma=0.05, so beta = R0*gamma/(<k^2>-<k>)/<k>.
# 3. Set initial condition: N=1000, infect 3 random nodes (rest susceptible, R=0).

import numpy as np
import scipy.sparse as sparse
import os

# Load network and parameters
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]

# Epidemiological parameters
R0 = 2.6
mean_k = 3.992
second_moment = 44.752
q = (second_moment - mean_k) / mean_k  # mean excess degree

gamma = 0.05 # recovery rate, as in several SIR/BA studies
beta = R0 * gamma / q

# Initial condition: randomly infect 3 nodes, the rest susceptible (I=3, S=997, R=0)
initial_conditions = {'S': N-3, 'I': 3, 'R': 0}

(beta, gamma, initial_conditions, q)

# Now model configuration: pick SIR (for COVID-like, representative for network), based on ER network first, SEIR can be handled similarly.
# Let's use the metrics from the previous step for the ER network
import numpy as np
N = 1000  # As set in previous network code

# Disease: COVID-19-like (fast-evolving, SIR is a first, SEIR can be next step if time)
# Use R0 = 3.0 (representative of COVID-19 early pandemic)
R0 = 3.0
mean_degree_er = 10.014  # From metrics
second_moment_er = 110.164

# For SIR on static network: beta = R0 * gamma / q, q = (<k^2> - <k>)/<k>
gamma = 0.1  # Recovery rate (10-day infectious period)
q = (second_moment_er - mean_degree_er) / mean_degree_er
beta = R0 * gamma / q

params = {'beta': float(beta), 'gamma': float(gamma), 'mean_degree_er': mean_degree_er, 'second_moment_er': second_moment_er, 'q': float(q), 'R0': R0 }

# Initial condition: 990 susceptible, 10 infected, 0 removed
initial_conditions = {'S': 990, 'I': 10, 'R': 0}

params, initial_conditions