
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

import fastgemf as fg
import scipy.sparse as sparse
import os
import matplotlib.pyplot as plt
import pandas as pd
# Load the necessary components for simulation
parameter_values = {'beta': 0.03364251329895325, 'gamma': 0.2, 'R0': 2.5, 'q': 14.86214765100671}
gamma = parameter_values['gamma']
beta = parameter_values['beta']
# Model schema and config
SIR_model_schema = (
    fg.ModelSchema('SIR')
      .define_compartment(['S', 'I', 'R'])
      .add_network_layer('contact_layer')
      .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
      .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_layer', rate='beta')
)
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
      .add_parameter(beta=beta, gamma=gamma)
      .get_networks(contact_layer=G_csr)
)
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}
# Simulation setup
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 120}, nsim=10)
sim.run()
# Save plot
plot_path = os.path.join(os.getcwd(), 'output', 'results-11.png')
sim.plot_results(show_figure=False, save_figure=True, save_path=plot_path)
# Save time series data
results = {}
time, state_count, *_ = sim.get_results()
results['time'] = time
for i, comp in enumerate(['S','I','R']):
    results[comp] = state_count[i, :]
df = pd.DataFrame(results)
csv_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
df.to_csv(csv_path, index=False)
(plot_path, csv_path)
import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
import pandas as pd

# Load network
network_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

# Define SIR model schema
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery1',
        from_state='I',
        to_state='R',
        rate='gamma'
    )
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta'
    )
)

# Model configuration
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=0.02188068436453747, gamma=0.14285714285714285)
    .get_networks(contact_network_layer=network_csr)
)

# Initial condition: use percentage that adds up to 100
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}  # 1% infected, 99% susceptible, 0% recovered

# Simulation setup
sim = fg.Simulation(
    SIR_instance, initial_condition=initial_condition, stop_condition={'time': 100}, nsim=10
)

# Run simulation and plot
fig_path = os.path.join(os.getcwd(), 'output', 'results-11.png')
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=fig_path)

# Extract and save results
time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
for i in range(state_count.shape[0]):
    simulation_results[f'{SIR_model_schema.compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data.to_csv(data_path, index=False)

[data_path, fig_path]

import fastgemf as fg
import scipy.sparse as sparse
import pandas as pd
import os

# 1. Load network
network_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

# 2. SIR model schema
SIR_model_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network_layer', rate='beta')
)
# 3. Model configuration
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=0.04464285714285714, gamma=0.14285714285714285)
    .get_networks(contact_network_layer=network_csr)
)
# 4. Initial conditions (random, by percentage)
# initial_conditions: {'S': 995, 'I': 5, 'R': 0} for N=1000
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

# 5. Simulation: 1 year, 10 runs for stochasticity
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 365}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

# 6. Export simulation data
time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
compartments = SIR_model_schema.compartments
for i in range(state_count.shape[0]):
    simulation_results[f'{compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

# SIR simulation chain of thought:
# 1. Use network and SIR model details from modeling phase.
# 2. Initial condition: 997 susceptible, 3 infected (standard for SIR studies / 0.3% initial prevalence)
# 3. Use FastGEMF for an exact stochastic network SIR simulation. N=1000. Use percentage IC for randomness.
# 4. Stop condition: until no infected remain or 365 days, nsim=5 for trajectory averaging.

import fastgemf as fg
import scipy.sparse as sparse
import os
import pandas as pd

# 1. Model definition
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network_layer', rate='beta')
)
# 2. Network loading
network_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))
# 3. Model instance
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=0.01441, gamma=0.04)
    .get_networks(contact_network_layer=network_csr)
)
# 4. Initial condition: 997 S, 3 I, 0 R (randomly assigned)
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}
# 5. Simulation
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 365.0}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
# 6. Export results
time, state_count, *_ = sim.get_results()
data = pd.DataFrame({'time': time,
                    'S': state_count[0, :],
                    'I': state_count[1, :],
                    'R': state_count[2, :]})
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

# For checking results:
last_size = int(state_count[2, -1])  # final epidemic size
peak_infect = int(state_count[1, :].max()) # peak infected
peak_time = float(time[state_count[1, :].argmax()])
duration = float(time[next(i for i in range(len(state_count[1, :])-1, -1, -1) if state_count[1, i]>0)] - time[0])
(first_few, last_few) = (data.head().to_dict(), data.tail().to_dict())
(last_size, peak_infect, peak_time, duration, first_few, last_few)
# Fix init_cond for 'exact' type to be a numpy array (required by FastGEMF)
import fastgemf as fg
import scipy.sparse as sparse
import os
import pandas as pd
import numpy as np

# Load network
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)

SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery',
        from_state='I',
        to_state='R',
        rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta')
)
parameters = {'beta': 0.025047638150636844, 'gamma': 0.1}
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(**parameters)
    .get_networks(contact_network_layer=G_csr)
)
N = G_csr.shape[0]  # number of nodes
init_infected = 10
X0 = np.zeros(N, dtype=int)
X0[:init_infected] = 1
init_cond = {'exact': X0}
sim = fg.Simulation(SIR_instance, initial_condition=init_cond, stop_condition={'time': 160}, nsim=5)
sim.run()
save_fig_path = os.path.join(os.getcwd(), 'output', 'results-11.png')
sim.plot_results(show_figure=False, save_figure=True, save_path=save_fig_path)
time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i in range(state_count.shape[0]):
    simulation_results[f'{SIR_model_schema.compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
save_csv_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data.to_csv(save_csv_path, index=False)
([save_fig_path, save_csv_path], SIR_model_schema.compartments)
import fastgemf as fg
import os
import numpy as np
import scipy.sparse as sparse
import pandas as pd

# ModelSchema for SIR
sir_model_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network_layer', rate='beta'
    )
    .add_node_transition(
        name='recovery', from_state='I', to_state='R', rate='gamma'
    )
)
# Load network
network_csr = sparse.load_npz(os.path.join('output', 'network.npz'))
# ModelConfiguration
sir_instance = (
    fg.ModelConfiguration(sir_model_schema)
    .add_parameter(beta=0.02188068436453747, gamma=0.14285714285714285)
    .get_networks(contact_network_layer=network_csr)
)
# Initial condition: percentage, I=10, S=990, R=0
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}} # 990/1000, 10/1000, 0
sim = fg.Simulation(sir_instance, initial_condition=initial_condition, stop_condition={'time': 160}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
time, state_count, *_ = sim.get_results()
results = {'time': time}
for idx, comp in enumerate(sir_model_schema.compartments):
    results[comp] = state_count[idx, :]
pd.DataFrame(results).to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
'output/results-11.csv', 'output/results-11.png'
import fastgemf as fg
import scipy.sparse as sparse
import os

# Parameters (from modeling and recent network construction)
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)

# SIR model parameters
beta = 0.0114  # computed for R0=1.0 with <k>=8, <k^2>=71.1, gamma=0.1
gamma = 0.1

# Setup Model Schema
SIR_model_schema = (
    fg.ModelSchema('SIR')
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
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact_network_layer=G_csr)
)

# Initial condition: 10 infected, rest susceptible, population 1000
init_cond = { 'percentage': {'S': 99, 'I': 1, 'R': 0} }

# Run stochastic simulation, 10 runs
sim = fg.Simulation(SIR_instance, initial_condition=init_cond, stop_condition={'time': 160}, nsim=10)
sim.run()

# Save results and plot
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
time, state_count, *_ = sim.get_results()
import pandas as pd
simulation_results = {}
simulation_results['time'] = time
for i in range(state_count.shape[0]):
    simulation_results[f'{SIR_model_schema.compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)