
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
import fastgemf as fg
from scipy import sparse
import os

# Load network
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))
N = G_csr.shape[0]

# Reproduce model schema and configuration as above
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta')
)
SIR_config = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=0.07763471110149317, gamma=0.25)
    .get_networks(contact_network_layer=G_csr)
)
# Initial condition: 10 infected, rest susceptible
init_cond = {'percentage': {'S': 99, 'I': 1, 'R': 0}}
# Simulate for 365 days, multiple stochastic runs
sim = fg.Simulation(SIR_config, initial_condition=init_cond, stop_condition={'time': 365}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
# Save CSV of one of the runs
import pandas as pd
time, state_count, *_ = sim.get_results()
sim_results = {'time': time}
for i, c in enumerate(['S', 'I', 'R']):
    sim_results[c] = state_count[i, :]
df_out = pd.DataFrame(sim_results)
df_out.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import random

# -- RANDOM VACCINATION SIMULATION --
R0 = 4
mean_degree = 2.97  # from previous simulation
k2 = 11.806         # from previous simulation
N = 5000            # number of nodes
beta = None
recover_rate = 1.0  # per-unit time
q = (k2 - mean_degree) / mean_degree
beta = R0 * recover_rate / q

# Path to network
network_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_sim.npz'))

# Calculate random vaccination threshold (from analytics: 1 - 1/R0 = 0.75)
p_rand = 0.75

# Vaccinate random sample
V = int(np.round(N * p_rand))
all_nodes = np.arange(N)
vaccinated_nodes = np.random.choice(all_nodes, size=V, replace=False)

# Set up initial conditions with vaccinated nodes: state 2 (R, immune)
X0 = np.zeros(N, dtype=int)  # everyone starts S (0)
N_inf_init = 30
avail_nodes = list(set(all_nodes) - set(vaccinated_nodes))
I_nodes = np.random.choice(avail_nodes, size=N_inf_init, replace=False)
X0[I_nodes] = 1  # infect sufficient un-vaccinated randoms
X0[vaccinated_nodes] = 2  # vaccinated are Removed

initial_condition = {'exact': X0}

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
    .add_parameter(beta=beta, gamma=recover_rate)
    .get_networks(contact_network_layer=network_csr)
)
time_limit = 60
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': time_limit}, nsim=3)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
time, state_count, *_ = sim.get_results()
import pandas as pd
results = {'time': time}
results['S'] = state_count[0, :]
results['I'] = state_count[1, :]
results['R'] = state_count[2, :]
data = pd.DataFrame(results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
data
import os
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd

# Load network
network_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

# SIR Model schema
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

# Model parameters
beta = 0.050411335475916155
gamma = 0.2
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact_network_layer=network_csr)
)

initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

# Simulation
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 60}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i, c in enumerate(SIR_model_schema.compartments):
    simulation_results[f'{c}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

import fastgemf as fg
import scipy.sparse as sparse
import os
import pandas as pd

# Load SIR model schema
def get_sir_model_schema():
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
            name='infection',
            from_state='S',
            to_state='I',
            inducer='I',
            network_layer='contact_network_layer',
            rate='beta'
        )
    )
    return SIR_model_schema

# Load the network
current_dir = os.getcwd()
network_path = os.path.join(current_dir, 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)

# Set up the model parameters
beta = 0.0421
gamma = 0.25

schema = get_sir_model_schema()

SIR_instance = (
    fg.ModelConfiguration(schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact_network_layer=G_csr)
)

# Initial condition as percentages
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

# Configure and run the simulation
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 120}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(current_dir, 'output', 'results-11.png'))

# Save time-series data
(time, state_count, *_) = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
for i, comp in enumerate(schema.compartments):
    simulation_results[comp] = state_count[i, :]

data = pd.DataFrame(simulation_results)
csv_path = os.path.join(current_dir, 'output', 'results-11.csv')
data.to_csv(csv_path, index=False)
csv_path

import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os

# Model creation: SI1SI2S (exclusive competitive SIS)
competitive_SIS_schema = (
    fg.ModelSchema('competitive_SIS')
    .define_compartment(['S','I1','I2'])  # Susceptible, Infected by 1, Infected by 2
    .add_network_layer('layer_A')
    .add_network_layer('layer_B')
    .add_node_transition(
        name='rec1', from_state='I1', to_state='S', rate='delta1')
    .add_node_transition(
        name='rec2', from_state='I2', to_state='S', rate='delta2')
    .add_edge_interaction(
        name='inf1', from_state='S', to_state='I1', inducer='I1', network_layer='layer_A', rate='beta1')
    .add_edge_interaction(
        name='inf2', from_state='S', to_state='I2', inducer='I2', network_layer='layer_B', rate='beta2')
)

# Load networks
A_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'layer_A-01.npz'))
B_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'layer_B-01.npz'))

params = {'beta1': 0.09973148145995928, 'delta1': 1.0, 'beta2': 0.09682856412454038, 'delta2': 1.0}
competitive_instance = (
    fg.ModelConfiguration(competitive_SIS_schema)
    .add_parameter(**params)
    .get_networks(layer_A=A_csr, layer_B=B_csr)
)

# Initial condition: start both viruses at low prevalence, 10 each on random nodes, all others susceptible
N = A_csr.shape[0]
X0 = np.zeros(N, dtype=int)  # S=0, I1=1, I2=2
I1_indices = np.random.choice(np.arange(N), size=10, replace=False)
X0[I1_indices] = 1
remain = list(set(range(N)) - set(I1_indices))
I2_indices = np.random.choice(remain, size=10, replace=False)
X0[I2_indices] = 2
initial_condition = {'exact': X0}

# Create simulation object
sim = fg.Simulation(competitive_instance, initial_condition=initial_condition, stop_condition={'time': 200}, nsim=8)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
# Save the simulated data
import pandas as pd
time, state_count, *_ = sim.get_results()
sim_data = {'time': time}
for i, comp in enumerate(competitive_SIS_schema.compartments):
    sim_data[comp] = state_count[i,:]
df = pd.DataFrame(sim_data)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
# Parameters from analytic step
beta1 = 0.45
delta1 = 0.18
beta2 = 0.32
delta2 = 0.12
N = 100
# Load networks
A = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_layerA.npz'))
B = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_layerB.npz'))
# Build competitive SIS model (susceptible, infected by 1, infected by 2), exclusive, two contact layers
class CompetitiveSIS:
    def __init__(self):
        self.schema = (
            fg.ModelSchema("SIS2Exclusive")
            .define_compartment(['S', 'I1', 'I2'])
            .add_network_layer('LayerA').add_network_layer('LayerB')
            .add_edge_interaction(name='inf1', from_state='S', to_state='I1', inducer='I1', network_layer='LayerA', rate='beta1')
            .add_edge_interaction(name='inf2', from_state='S', to_state='I2', inducer='I2', network_layer='LayerB', rate='beta2')
            .add_node_transition(name='rec1', from_state='I1', to_state='S', rate='delta1')
            .add_node_transition(name='rec2', from_state='I2', to_state='S', rate='delta2')
        )
        self.config = (
            fg.ModelConfiguration(self.schema)
            .add_parameter(beta1=beta1, delta1=delta1, beta2=beta2, delta2=delta2)
            .get_networks(LayerA=A, LayerB=B)
        )
# Initial condition: 7% infected by 1, 7% infected by 2, rest S, random
X0 = np.zeros(N, dtype=int) # all S
X0[:7] = 1  # 7 infected by 1 (I1)
X0[7:14] = 2  # 7 infected by 2 (I2)
np.random.seed(42)
np.random.shuffle(X0)
init_cond = {'exact': X0}
model = CompetitiveSIS()
sim = fg.Simulation(model.config, initial_condition=init_cond, stop_condition={'time': 200}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
import pandas as pd
time, state_count, *_ = sim.get_results()
table = {'time': time, 'S': state_count[0, :], 'I1': state_count[1, :], 'I2': state_count[2, :]}
data = pd.DataFrame(table)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

# Given results from network construction, now:
# 1. Set SI1SI2S (competitive SIS) model schema for two exclusive viruses on two layers.
# 2. Set parameters: choose beta1, delta1 for layer A and beta2, delta2 for layer B such that both effective infection rates are above the mean-field (NIMFA) threshold for their respective layers.
# 3. Initial condition: randomly infect 10% of nodes for each virus, rest susceptible.
# 4. Save code to file.

import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import os

N = 1000

# Load networks
networkA_csr = sparse.load_npz('/Users/hosseinsamaei/phd/gemf_llm/output/networkA.npz')
networkB_csr = sparse.load_npz('/Users/hosseinsamaei/phd/gemf_llm/output/networkB.npz')

# Mean degrees
mean_deg_A = 7.968
mean_deg2_A = 143.698
mean_deg_B = 7.824
mean_deg2_B = 131.552

# SIS-like competitive SI1SI2S model:
# Effective infection rate tau1 = beta1/delta1, tau2 = beta2/delta2
# NIMFA threshold: tau_c = 1/lambda1(A), lambda1(A) ≈ mean_deg2/mean_deg

lambda1A = mean_deg2_A/mean_deg_A
lambda1B = mean_deg2_B/mean_deg_B

# Select effective infection rates above threshold
# e.g., tau1 = 1.5/lambda1A, tau2 = 1.5/lambda1B
beta1 = 0.15  # Layer A (virus 1)
delta1 = beta1 / (1.5/lambda1A)
beta2 = 0.13  # Layer B (virus 2)
delta2 = beta2 / (1.5/lambda1B)

# Model definition
gemf_model = (
    fg.ModelSchema('SI1SI2S')
    .define_compartment(['S', 'I1', 'I2'])
    .add_network_layer('A')
    .add_network_layer('B')
    .add_node_transition(name='recover1', from_state='I1', to_state='S', rate='delta1')
    .add_node_transition(name='recover2', from_state='I2', to_state='S', rate='delta2')
    .add_edge_interaction(name='infect1', from_state='S', to_state='I1', inducer='I1', network_layer='A', rate='beta1')
    .add_edge_interaction(name='infect2', from_state='S', to_state='I2', inducer='I2', network_layer='B', rate='beta2')
)

# Parameterize, assign network layers
config = (
    fg.ModelConfiguration(gemf_model)
    .add_parameter(beta1=beta1, delta1=delta1, beta2=beta2, delta2=delta2)
    .get_networks(A=networkA_csr, B=networkB_csr)
)

# Initial condition: random 10% I1, 10% I2, rest S, but ensure exclusivity (cannot overlap)
X0 = np.zeros(N, dtype=int)  # all S (0)
idx = np.arange(N)
np.random.shuffle(idx)
I1_idx = idx[:N//10]
I2_idx = idx[N//10:N//5]
X0[I1_idx] = 1
X0[I2_idx] = 2
initial_condition = {'exact': X0}

# Run simulation
sim = fg.Simulation(config, initial_condition=initial_condition, stop_condition={'time': 250}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

# Save simulation results to CSV
import pandas as pd
time, state_count, *_ = sim.get_results()
gemf_results = {'time': time}
for i, cname in enumerate(['S', 'I1', 'I2']):
    gemf_results[cname] = state_count[i,:]
data = pd.DataFrame(gemf_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

# For reproducibility, save parameter info
sim_info = {
    'beta1': beta1, 'delta1': delta1, 'beta2': beta2, 'delta2': delta2,
    'lambda1A': lambda1A, 'lambda1B': lambda1B,
    'tau1': beta1/delta1, 'tau2': beta2/delta2
}

sim_info
import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
import pandas as pd

# Load network and IC
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
X0 = np.load(os.path.join(os.getcwd(), 'output', 'X0.npy'))

# Model schema
SIR_model_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network_layer', rate='beta')
)
G_csr = sparse.load_npz(network_path)
params = {'beta': 0.004179573493203892, 'gamma': 0.04}
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(**params)
    .get_networks(contact_network_layer=G_csr)
)
initial_condition = {'exact': X0}
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 365}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i, state in enumerate(['S', 'I', 'R']):
    simulation_results[state] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
output = {
    'csv_path': os.path.join(os.getcwd(), 'output', 'results-11.csv'),
    'png_path': os.path.join(os.getcwd(), 'output', 'results-11.png')
}
output
# SIR model over Barabasi-Albert network
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import os
import pandas as pd

# Model parameters inferred from basic values in literature and the network structure
R0 = 2.5  # Typical value for moderately infectious disease
recovery_rate = 0.1  # gamma
# Get degree moments (inserted from previous output)
mean_k = 7.968
second_moment_k = 144.722
# For SIR on networks: q = (<k^2> - <k>) / <k>
q = (second_moment_k - mean_k) / mean_k
# beta = R0 * gamma / q
beta = R0 * recovery_rate / q

# Paths (replace with actual paths from previous step)
output_dir = os.path.join(os.getcwd(), 'output')
network_path = os.path.join(output_dir, 'network.npz')

G_csr = sparse.load_npz(network_path)

# Model schema: SIR
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
        network_layer='contact_network_layer', rate='beta')
)

SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=beta, gamma=recovery_rate)
    .get_networks(contact_network_layer=G_csr)
)

# Initial conditions: Random initialization with 10 infected, rest susceptible
N = G_csr.shape[0]
I0 = 10
S0 = N - I0
R0c = 0
initial_condition = {'percentage': {'S': int(100 * S0 / N), 'I': int(100 * I0 / N), 'R': 0}}

sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 100}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(output_dir, 'results-11.png'))
time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i, comp in enumerate(SIR_model_schema.compartments):
    simulation_results[comp] = state_count[i, :]
pd.DataFrame(simulation_results).to_csv(os.path.join(output_dir, 'results-11.csv'), index=False)

# End with the beta used for reproducibility
beta, recovery_rate, initial_condition

import fastgemf as fg
import scipy.sparse as sparse
import pandas as pd
import numpy as np
import os
# --- SIR model schema ---
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network', rate='beta')
)
# --- Load ER network ---
net_path = os.path.join(os.getcwd(), 'output', 'network_ER.npz')
G_csr = sparse.load_npz(net_path)
# --- Set parameters for ER ---
params = {'beta': 0.06235, 'gamma': 0.2}
SIR_model = fg.ModelConfiguration(SIR_schema).add_parameter(**params).get_networks(contact_network=G_csr)
# --- Initial condition (random) ---
ic = {'percentage': {'S': 99, 'I': 1, 'R': 0}}
# --- Run simulation ---
sim = fg.Simulation(SIR_model, initial_condition=ic, stop_condition={'time': 120}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
time, state_count, *_ = sim.get_results()
data = {'time': time}
for i, comp in enumerate(SIR_schema.compartments):
    data[comp] = state_count[i, :]
df = pd.DataFrame(data)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os

# Define the SIR model schema for network-based epidemic spread
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

# Compute infection and recovery rates given R0 and network moments (for SIR model over network):
# beta = R0 * gamma / q; q = (<k^2> - <k>) / <k>
mean_deg = 9.97
second_moment_deg = 108.786
q = (second_moment_deg - mean_deg) / mean_deg
R0 = 2.5  # Representative for moderately infectious disease (such as COVID-19, no interventions)
gamma = 1/7  # Recovery rate: avg. infectious period is 7 days (gamma=0.143)
beta = R0 * gamma / q

# Load network
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)

# Model configuration
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=float(beta), gamma=float(gamma))
    .get_networks(contact_network_layer=G_csr)
)

# Random initial condition: 10 infected, all others susceptible (for 1000 nodes)
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

# Create output directory if it does not exist
os.makedirs(os.path.join(os.getcwd(), 'output'), exist_ok=True)

# Run simulation: 365 days, 5 replications
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 365}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

# Save results to CSV
import pandas as pd
time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i, name in enumerate(SIR_model_schema.compartments):
    simulation_results[name] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

# Return beta, gamma, q, network path, initial condition
dict(beta=beta, gamma=gamma, q=q, network_path=network_path, initial_condition=initial_condition)