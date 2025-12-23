
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
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os

output_dir = os.path.join(os.getcwd(), 'output')

# Load network layers
G_ER = sparse.load_npz(os.path.join(output_dir, 'network-ER.npz'))
G_BA = sparse.load_npz(os.path.join(output_dir, 'network-BA.npz'))

# Model: SIR model (Susceptible, Infected, Recovered)
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network')
    .add_node_transition(
        name='recovery',
        from_state='I',
        to_state='R',
        rate='gamma'
    )
    .add_edge_interaction(
        name='infection',
        from_state='S',
        to_state='I',
        inducer='I',
        network_layer='contact_network',
        rate='beta'
    )
)

# Parameters
N = 1000
R0 = 2.5                  # Typical for moderately-contagious viruses
mean_k_ER = 8.036
mean_k2_ER = 72.48
mean_k_BA = 7.968
mean_k2_BA = 138.024

gamma = 1/7               # 1 week infectious period (days)
# For network models: beta = R0*gamma / ((<k2> - <k>)/<k>)
def compute_beta(R0, gamma, mean_k, mean_k2):
    q = (mean_k2 - mean_k) / mean_k
    return R0 * gamma / q
beta_ER = compute_beta(R0, gamma, mean_k_ER, mean_k2_ER)
beta_BA = compute_beta(R0, gamma, mean_k_BA, mean_k2_BA)

# Model configurations
SIR_ER = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta_ER, gamma=gamma)
    .get_networks(contact_network=G_ER)
)
SIR_BA = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta_BA, gamma=gamma)
    .get_networks(contact_network=G_BA)
)

# Initial condition - 10 infected randomly, rest susceptible
initial_cond = {'percentage': {'I': 1, 'S': 99, 'R': 0}}

# Store for appendices
parameters_info = {
    'R0': R0,
    'gamma': gamma,
    'beta_ER': beta_ER,
    'beta_BA': beta_BA
}

initial_cond_info = 'Random initial infection: 10 individuals infected, remainder susceptible. No recovered at t=0.'


# Definition of the SIR model schema for network-based simulation
import fastgemf as fg
from scipy import sparse
import os

# Model Schema: SIR
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery',
        from_state='I',
        to_state='R',
        rate='gamma'
    )
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta'
    )
)

# Load precomputed network
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]
# Parameter Setup: for COVID-19 with network, literature-based for static network SIR
R0 = 3.0  # R0 for COVID-19 early in pandemic, conservative value from literature
mean_k = 7.83
second_k = 69.398
gamma = 1/14  # Recovery rate per day (infectious period of 14 days typical)
q = (second_k - mean_k)/mean_k
beta = R0 * gamma / q  # Adjusted transmission rate for network structure
# Initial condition: 10 initial infections, rest susceptible, no recovered
initial_infected = 10
S = N - initial_infected
I = initial_infected
R = 0
initial_condition = {'percentage': {'S': round(100*S/N), 'I': round(100*I/N), 'R': round(100*R/N)}}
# Save schema parameters for reporting
model_params = {'R0': R0, 'gamma': gamma, 'beta': beta, 'mean_k': mean_k, 'second_k': second_k, 'q': q}
(model_params, initial_condition)

import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
# Set model parameters for SIR over network
R0 = 2.5  # basic reproduction number as relevant for COVID-19 case studies (can be varied in sensitivity analyses)
gamma = 0.2    # mean infectious period of 5 days (gamma = 1/mean infectious period), time unit is days
# Mean degree <k> and <k²> from earlier calculation:
mean_k = 14.9
mean_k2 = 236.346
# Compute mean excess degree q for SIR:
q = (mean_k2 - mean_k) / mean_k
beta = R0 * gamma / q
# Setup SIR ModelSchema
SIR_model_schema = (
    fg.ModelSchema('SIR')
      .define_compartment(['S', 'I', 'R'])
      .add_network_layer('contact_layer')
      .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
      .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_layer', rate='beta')
)
# Load the ER network
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)
# Initial condition: random, start with 10 initial infections
N = 1000
I0 = 10
S0 = N - I0
R0_init = 0
initial_condition = {'percentage': {'S': int(100*S0/N), 'I': int(100*I0/N), 'R': 0}} # random allocation
# Model instance configuration
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
      .add_parameter(beta=beta, gamma=gamma)
      .get_networks(contact_layer=G_csr)
)
# Save parameter values (for reporting/analysis)
parameter_values = {'beta': beta, 'gamma': gamma, 'R0': R0, 'q': q}
# Save code as parameter_setting.py
import pickle
with open(os.path.join(os.getcwd(), 'output', 'parameter_setting.pkl'), 'wb') as f:
    pickle.dump(parameter_values, f)
(parameter_values, initial_condition)
# SIR model parameter calculation
# R0 (basic reproduction number) is assumed to be typical for COVID-19-like disease, e.g. R0=2.5
# Recovery rate gamma corresponds to average infectious period of 7 days (gamma = 1/7)

import numpy as np

R0 = 2.5
infectious_period = 7  # days
n = 1000  # as set in network construction
mean_k = 7.968
second_moment_k = 138.024

# For SIR model on static networks, beta = R0 * gamma / mean excess degree (q)
gamma = 1 / infectious_period
q = (second_moment_k - mean_k) / mean_k
beta = R0 * gamma / q

# Initial conditions: 10 nodes infected, rest susceptible (no recovered)
init_infected = 10
init_susceptible = n - init_infected
init_recovered = 0
initial_condition = {'S': init_susceptible, 'I': init_infected, 'R': init_recovered}
initial_condition_type = 'percentage'

# Save parameters and initial conditions for record
with open('output/parameter_setting.py', 'w') as f:
    f.write(f"""
SIR model parameter setting for Barabasi-Albert network
R0 = {R0}
gamma = {gamma}
beta = {beta}
mean_k = {mean_k}
second_moment_k = {second_moment_k}
Initial condition: {initial_condition}
Initial condition type: {initial_condition_type}
""")

(beta, gamma, initial_condition, initial_condition_type)
import os
# SIR model: Parameter setting
N = 1000
# Based on literature (COVID-19, respiratory disease):
R0 = 2.5  # Typical for COVID-19 on contact networks [see literature]
gamma = 1/7  # Recovery rate ~1 week infectious period
# Estimation for ER network beta:
# Mean degree, <k> = 8, Second moment <k2> as needed, but for ER: <k2> = <k>^2 + <k>
mean_k = 8
second_moment = mean_k**2 + mean_k  # for ER
q = (second_moment - mean_k) / mean_k
beta = R0 * gamma / q
# Initial conditions: 995 S, 5 I, 0 R (random distribution)
initial_conditions = {"S": 995, "I": 5, "R": 0}
return_vars = ["beta", "gamma", "initial_conditions"]
# Parameters and initial condition configuration for SIR model adapted for static network
import math
N = 1000
init_infected = 10
init_recovered = 0
init_susceptible = N - init_infected - init_recovered
# R0 is typically 2.5 for moderately contagious diseases (e.g., COVID-19-like)
R0 = 2.5
# mean_degree and second_moment_degree will be inserted after network construction
mean_degree = 9.99  # placeholder, update after network construction
second_moment_degree = 109.7  # placeholder, update after network construction
# Compute q for ER network
q = (second_moment_degree - mean_degree) / mean_degree
# Set recovery rate (gamma), e.g., 0.1 (mean infectious period = 10)
gamma = 0.1
# Compute infection rate beta for SIR on network
beta = R0 * gamma / q
parameters = {'beta': beta, 'gamma': gamma}
initial_conditions = {'S': init_susceptible, 'I': init_infected, 'R': init_recovered}
(parameters, initial_conditions, q, R0, mean_degree, second_moment_degree)
# SIR model simulation parameter setting
import numpy as np
import os
import json

# Known Parameters
R0 = 2.5  # representative for moderate infectious disease
recovery_rate = 1/7  # 1 week infectious period, gamma = 0.142857
# Network properties (from modeling phase)
nodes = 1000
mean_deg = 7.968
mean_deg2 = 138.024
# SIR on network: mean excess degree q = (<k^2> - <k>) / <k>
q = (mean_deg2 - mean_deg) / mean_deg
# Transmission rate beta adjusted for network
beta = R0 * recovery_rate / q
# Initial condition: 10 infected (random), rest susceptible
I0 = 10
S0 = nodes - I0
R0_pop = 0
init_cond = {'S':int(S0),'I':int(I0),'R':int(R0_pop)}
# Save the scenario details for reproducibility
scenario = {
    'R0': R0,
    'recovery_rate': recovery_rate,
    'beta': beta,
    'init_cond': init_cond,
    'network': {
        'mean_deg': float(mean_deg),
        'mean_deg2': float(mean_deg2),
        'q': float(q),
        'nodes': nodes
    }
}
with open(os.path.join('output','sir_parameters.json'),'w') as f:
    json.dump(scenario,f,indent=2)
beta, recovery_rate, init_cond
import fastgemf as fg
from scipy import sparse
import numpy as np
import os

# 1. Model schema: SIR on contact network
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery', from_state='I', to_state='R', rate='gamma'
    )
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta'
    )
)
# 2. Load network
net_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(net_path)
# 3. Parameters based on desired R0 and network stats
R0 = 2.5  # moderate-high transmission, realistic for diseases such as COVID-19 at baseline
gamma = 0.25  # per day recovery (average infectious period = 4 days)
mean_k = 8.036
mean_k2 = 72.73
q = (mean_k2 - mean_k) / mean_k
beta = R0 * gamma / q
# 4. Configuration
SIR_config = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact_network_layer=G_csr)
)
# 5. Initial condition: 10 initial infecteds, rest susceptible
N = G_csr.shape[0]
I0 = 10
S0 = N - I0
R0_num = 0
init_cond = {'percentage': {'S': int(round(S0*100/N)), 'I': int(round(I0*100/N)), 'R': 0}}
# 6. Save all settings for documentation
import json
with open(os.path.join('output', 'model_parameters.json'), 'w') as f:
    json.dump({'beta': beta, 'gamma': gamma, 'R0': R0, 'network': 'Erdos-Renyi',
               'mean_k': mean_k, 'mean_k2': mean_k2}, f)
# Print important params
print('beta', beta)
print('gamma', gamma)
print('init_cond', init_cond)
# SIR model mechanistic setup with parameters
# Assume beta=0.5 and gamma=0.2 (1/gamma=5 days), typical values for moderate infectivity diseases
# Let's adjust beta for network, using R0=beta/gamma * <k^2-<k>>/<k> for SIR on network (mean field).
# For ER, <k> ~ 10, <k^2> ~ 108.9
# Assume R0=2.5 as typical
mean_degree = 9.976
second_moment = 108.922
R0 = 2.5
gamma = 0.2  # Recovery rate, 1/gamma = 5 days
# mean excess degree (q)
q = (second_moment - mean_degree) / mean_degree
beta = R0 * gamma / q
# Initial condition: N=1000, 10 infected, rest susceptible
N = 1000
I0 = 10
S0 = N - I0
R0_pop = 0
initial_conditions = {'S': S0, 'I': I0, 'R': R0_pop}

# Compute SIR model network rates based on R_0, <k> and <k^2>
# Parameters extracted from sources: R_0 = 1.67, recovery period = 4 days
R0 = 1.67
gamma = 1/4  # Recovery rate (1/days), gamma, so ~0.25
mean_k = 9.97
second_moment_k = 108.786
mean_excess_degree = (second_moment_k - mean_k) / mean_k  # q
beta = R0 * gamma / mean_excess_degree  # Network-based infection rate

# Initial population setup
N = 1000
initial_infected = int(0.01 * N)  # 1% initially infected
initial_recovered = 0
initial_susceptible = N - initial_infected - initial_recovered

# Express as percent for fastgemf
initial_percent = {'S': round(initial_susceptible/N*100), 'I': round(initial_infected/N*100), 'R': round(initial_recovered/N*100)}
(beta, gamma, initial_percent)
# Calculation for SIR on BA network, setting beta/gamma rates for target R0, outputting model and parameter summary
N = 1000
init_S = 997
init_I = 3
init_R = 0
R0 = 3.0  # between 2-3 for COVID, classic value for demonstration
mean_k = 15.872
mean_k2 = 471.574
q = (mean_k2 - mean_k) / mean_k

gamma = 0.04  # 1/gamma = 25 days avg infectious period
beta = R0 * gamma / q  # network-adapted infection rate

model_schema = {
    'name': 'SIR',
    'compartments': ['S', 'I', 'R'],
    'transitions': {'S-(I)->I': 'beta', 'I->R': 'gamma'},
    'reasoning_info': 'SIR is chosen because it best fits a disease such as COVID-19 or classical influenza with recovery conferring immunity. The BA network mimics real-world heterogeneity.'
}
parameters = {'beta': float(beta), 'gamma': gamma}
initial_condition_desc = ['randomly assign 3 infected and 997 susceptible among 1000 nodes']
initial_condition_type = ['percentage']
initial_conditions = [{'S': 997, 'I': 3, 'R': 0}]
reasoning_info = 'Parameters based on typical COVID SIR values; initial infections are seeded randomly reflecting likely unknown patient-zero locations.'

results = dict(
    model_schema=model_schema,
    parameters=parameters,
    initial_condition_desc=initial_condition_desc,
    initial_condition_type=initial_condition_type,
    initial_conditions=initial_conditions,
    reasoning_info=reasoning_info,
    R0=R0,
    q=q
)
results
# Calculate beta for SIR on both networks given R0, gamma, <k>, <k^2>
R0 = 2.5
gamma = 0.2  # [1/days], average infectious period = 5 days
net_stats = {
    'ER': {'mean_k': 8.036, 'mean_k2': 72.48},
    'BA': {'mean_k': 7.968, 'mean_k2': 138.024},
}
# mean excess degree q = (<k^2> - <k>)/<k>
betas = {}
for key, s in net_stats.items():
    q = (s['mean_k2'] - s['mean_k'])/s['mean_k']
    betas[key] = R0 * gamma / q

# Initial condition: 10 infected, rest susceptible (N=1000), random
N = 1000
I0 = 10
S0 = N - I0
R0_init = 0
initial_condition_percent = {'S': int(100*S0/N), 'I': int(100*I0/N), 'R': int(100*R0_init/N)}
return_vars = ['betas', 'initial_condition_percent']

import numpy as np
# SIR parameters
gamma = 0.2  # recovery rate per contact per day (recovery in ~5 days)
R0 = 2.5      # basic reproduction number (from e.g., COVID-19 [Han25], [Ochab10], typical for SIR on contact networks)
# mean_k = 10, mean_k2: needs input from network construction
mean_k = 10  # placeholder - REPLACE AFTER NETWORK
mean_k2 = 0  # placeholder - REPLACE AFTER NETWORK
# effective beta for network (see literature, Newman's formula for SIR threshold):
# q = (<k^2> - <k>)/<k> ; beta = R0 * gamma / q

# (We must replace mean_k, mean_k2 below with the result from previous code)
def compute_beta(R0, gamma, mean_k, mean_k2):
    q = (mean_k2 - mean_k) / mean_k
    return R0 * gamma / q

# Calculate beta (fill-in mean_k2)
beta = compute_beta(R0, gamma, mean_k, mean_k2)
print('beta:', beta)

# initial conditions % (reflecting typical COVID-19 scenario):
N = 1000
S0 = 990  # 99% susceptible
I0 = 10   # 1% infected
R0_0 = 0
init_cond = {'S':S0,'I':I0,'R':R0_0}

# Mechanistic model: SIR over static ER network
# Parameters from mean degree, R0, recovery rate
import numpy as np
import os

# From network construction
mean_k = 9.97
mean_k2 = 108.786

# Standard SIR parameters (suitable for COVID-19 like):
R0 = 2.5   # basic reproduction number
recov_rate = 1/7  # gamma, 1/average infectious period (7 days)
# For SIR over network: beta = R0 * gamma / q
q = (mean_k2 - mean_k) / mean_k
beta = R0 * recov_rate / q

desc = [
    'Initial infected are randomly chosen 1% of network; rest susceptible; no recovered. Size=1000.',
    'Alternative: Infect top 5 hubs in degree, rest susceptible.'
]
# Convert to integers: 1% of 1000 = 10 infected, rest susceptible, recovered=0
init_ic = [ {'S': 990, 'I': 10, 'R': 0}, {'S': 995, 'I': 5, 'R': 0}]
initial_condition_type = [
    'percentage',
    'exact (hubs_number: top degree)'
]
params = {'beta': beta, 'gamma': recov_rate}
reasoning = (
    f"beta computed as R0 * gamma / q; ER network reflects homogeneous mixing." \
    f"Two scenarios: random seeding and targeting hubs; this allows assessing effect of initial conditions. "
)

output = {
    'parameters': params,
    'initial_condition_type': initial_condition_type,
    'initial_conditions': init_ic,
    'initial_condition_desc': desc,
    'reasoning_info': reasoning
}
output