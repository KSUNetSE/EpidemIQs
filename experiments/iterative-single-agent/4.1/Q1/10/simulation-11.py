
import fastgemf as fg
import numpy as np
import os
import pandas as pd
from scipy import sparse

# 1. Model schema
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(["S", "I", "R"])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network_layer', rate='beta')
)

# 2. Load network
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

# 3. Model config with parameters
sir_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=0.061012438703504375, gamma=0.2)
    .get_networks(contact_network_layer=G_csr)
)

# 4. Initial condition: corrected to sum to 100 per requirements
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}} # 1% infected, 99% susceptible

# 5. Simulation (up to 100 time units, 10 repetitions)
sim = fg.Simulation(sir_instance, initial_condition=initial_condition, stop_condition={'time': 100}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i, comp in enumerate(["S", "I", "R"]):
    simulation_results[comp] = state_count[i, :]
df = pd.DataFrame(simulation_results)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
import pandas as pd

# 1. Model Schema
schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_net')
    .add_edge_interaction('infection', from_state='S', to_state='I', inducer='I', network_layer='contact_net', rate='beta')
    .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
)

# 2. Load network
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

# 3. Model Configuration
params = {'beta': 0.03612045009495029, 'gamma': 0.142857}
config = (
    fg.ModelConfiguration(schema)
    .add_parameter(**params)
    .get_networks(contact_net=G_csr)
)

# 4. Initial condition (percentage)
init_cond = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

# 5. Simulate
sim = fg.Simulation(config, initial_condition=init_cond, stop_condition={'time': 120}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-11.png'))

# 6. Results
results = {}
time, state_count, *_ = sim.get_results()
results['time'] = time
for i, c in enumerate(schema.compartments):
    results[c] = state_count[i, :]
pd.DataFrame(results).to_csv(os.path.join(os.getcwd(),'output','results-11.csv'), index=False)

import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
# 1. Model schema
SIR_model_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network_layer', rate='beta')
)
# 2. Load the BA network
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))
# 3. Model configuration
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=0.012008119579258165, gamma=0.07)
    .get_networks(contact_network_layer=G_csr)
)
# 4. Initial condition: 99.7% S, 0.3% I, 0% R, rounded to nearest integer for a population of 1000
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}
# 5. Simulate, 1 realization
sim = fg.Simulation(
    SIR_instance,
    initial_condition=initial_condition,
    stop_condition={'time': 250},
    nsim=1
)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
# Save CSV
import pandas as pd
time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
for i in range(state_count.shape[0]):
    simulation_results[f'{SIR_model_schema.compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
import pandas as pd

# Load network
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)

# SIR model schema
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery', from_state='I', to_state='R', rate='gamma'
    )
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network_layer', rate='beta'
    )
)

# Set parameters
params = {'beta': 0.02999, 'gamma': 0.1}  # Used values from parameter_setting.py
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(**params)
    .get_networks(contact_network_layer=G_csr)
)

# Initial conditions by percentage, as calculated: 297 S, 3 I, 0 R
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

# Simulation settings
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 160}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
states = ['S', 'I', 'R']
for i in range(state_count.shape[0]):
    simulation_results[f'{states[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

import os
import numpy as np
import fastgemf as fg
import scipy.sparse as sparse

# 1. Model schema: SIR on contact network
SIR_model_schema = (
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
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

# 3. Parameters
R0 = 4.0
k_mean = 8.036
k2_mean = 72.48
q = (k2_mean - k_mean) / k_mean  # mean excess degree
gamma = 0.04  # recovery rate per day
beta = R0 * gamma / q  # inferred infection rate

# 4. Model configuration
sir_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact_network_layer=G_csr)
)

# 5. Initial conditions
N = G_csr.shape[0]
init_infected = max(int(0.005 * N), 1)  # at least 1 infected
X0 = np.zeros(N, dtype=int)
indices_infected = np.random.choice(N, size=init_infected, replace=False)
X0[indices_infected] = 1  # Set infected
initial_condition = {'exact': X0}

# 6. Simulation
sim = fg.Simulation(sir_instance, initial_condition=initial_condition, stop_condition={'time': 300}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

# 7. Save results to CSV
import pandas as pd
time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i in range(state_count.shape[0]):
    simulation_results[f'{SIR_model_schema.compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

# Report rates
beta, gamma, init_infected, N
# Fix: sum of percentage S+I+R must exactly add to 100 for FastGEMF
import fastgemf as fg
import os
import scipy.sparse as sparse
import numpy as np
import pandas as pd

# Load network
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]

# Model setup
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(["S", "I", "R"])
    .add_network_layer('contact_layer')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_layer', rate='beta')
)

SIR_instance = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=0.012732, gamma=0.05)
    .get_networks(contact_layer=G_csr)
)

# Set percentage so that S+I+R = 100 exactly
infected = 3
susceptible = N - infected
p_I = int(round(infected * 100 / N))  # 0.3% rounds to 0, so keep min 1% for at least a few nodes
p_S = 100 - p_I
init_cond = {'percentage': {'S': p_S, 'I': p_I, 'R': 0}}

# Safety: show assigned percentages and sum
assigned = (p_S, p_I, 0, p_S+p_I+0)

# Run simulation
sim = fg.Simulation(
    SIR_instance,
    initial_condition=init_cond,
    stop_condition={"time": 180},
    nsim=10
)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
time, state_count, *_ = sim.get_results()
results = {'time': time}
for i, state in enumerate(SIR_schema.compartments):
    results[state] = state_count[i, :]
df = pd.DataFrame(results)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

assigned # show assigned values for final review

import fastgemf as fg
import os
import numpy as np
import pandas as pd
from scipy import sparse

# 1. Model schema for SIR (over ER)
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('er_contact')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='er_contact', rate='beta')
    .add_node_transition(
        name='recovery', from_state='I', to_state='R', rate='gamma')
)

# 2. Load ER network
G_er_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'ER_network.npz'))

# 3. Create model configuration
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=0.029997004493260107, gamma=0.1)
    .get_networks(er_contact=G_er_csr)
)

# 4. Initial condition: correct percentage for 1000 nodes: 99% S, 1% I, 0% R (random nodes)
initial_percent = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

# 5. Simulate dynamics for 180 days, 5 stochastic runs
sim = fg.Simulation(SIR_instance, initial_condition=initial_percent, stop_condition={'time': 180}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
# Collect results to CSV
results_time, results_pop, *_ = sim.get_results()
out = {'time': results_time}
compartments = ['S', 'I', 'R']
for i in range(results_pop.shape[0]):
    out[compartments[i]] = results_pop[i, :]
pd.DataFrame(out).to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

import fastgemf as fg
import os
import scipy.sparse as sparse
import numpy as np
import pandas as pd
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]
# SIR Model Schema
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
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=0.027252097564597566, gamma=0.07142857142857142)
    .get_networks(contact_network_layer=G_csr)
)
# Initial condition for FastGEMF: 1% Infected (10), 99% Susceptible, 0% Recovered
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}
# Simulation setup
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 200}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
time, state_count, *_ = sim.get_results()
# Save all results to CSV
simulation_results = {}
simulation_results['time'] = time
for i in range(state_count.shape[0]):
    simulation_results[f'{SIR_model_schema.compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
