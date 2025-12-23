
# On review: very small fraction infected means all % go to S: need to use 'exact' mode for initial condition
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

# Initial condition: exact mode (map: S=0, I=1, R=2)
X0 = np.zeros(N, dtype=int)
I_nodes = np.random.choice(N, 3, replace=False)
X0[I_nodes] = 1
init_cond = {'exact': X0}

# Run simulation
sim = fg.Simulation(
    SIR_instance,
    initial_condition=init_cond,
    stop_condition={"time": 180},
    nsim=10
)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-12.png'))
time, state_count, *_ = sim.get_results()
results = {'time': time}
for i, state in enumerate(SIR_schema.compartments):
    results[state] = state_count[i, :]
df = pd.DataFrame(results)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-12.csv'), index=False)

# Output paths for reporting
['/Users/hosseinsamaei/phd/gemf_llm/output/results-12.csv','/Users/hosseinsamaei/phd/gemf_llm/output/results-12.png']

import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import random

# Parameters
N = 5000
z = 3.0
k2 = 11.806
R0 = 4
mean_degree = 2.97
recover_rate = 1.0
q = (k2 - mean_degree) / mean_degree
beta = R0 * recover_rate / q
network_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_sim.npz'))
G = None
try:
    import networkx as nx
    G = nx.from_scipy_sparse_array(network_csr)
except ImportError:
    G = None

# Find all nodes with degree exactly 10
if G is not None:
    degree_10_nodes = [n for n, d in G.degree() if d == 10]
else:
    degree_10_nodes = []
V = len(degree_10_nodes)
# For simulation: try with all degree-10 nodes vaccinated
X0 = np.zeros(N, dtype=int)  # all susceptible initially
N_inf_init = 30
all_nodes = np.arange(N)
avail_nodes = list(set(all_nodes) - set(degree_10_nodes))
# Infect 30 random nodes who are not in degree-10 set
I_nodes = np.random.choice(avail_nodes, size=min(N_inf_init, len(avail_nodes)), replace=False)
X0[I_nodes] = 1
for n in degree_10_nodes:
    X0[n] = 2  # vaccinated individuals (immune/removed)

init_cond = {'exact': X0}
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
sim = fg.Simulation(SIR_instance, initial_condition=init_cond, stop_condition={'time': time_limit}, nsim=3)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-12.png'))
time, state_count, *_ = sim.get_results()
import pandas as pd
results = {'time': time}
results['S'] = state_count[0, :]
results['I'] = state_count[1, :]
results['R'] = state_count[2, :]
data = pd.DataFrame(results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-12.csv'), index=False)
# Compute final fraction vaccinated
frac_vaccinated = len(degree_10_nodes)/N
# Save stats
stats = {'n_vaccinated': len(degree_10_nodes), 'frac_vaccinated': frac_vaccinated}
stats
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
# --- Load BA network ---
net_path = os.path.join(os.getcwd(), 'output', 'network_BA.npz')
G_csr = sparse.load_npz(net_path)
# --- Set parameters for BA ---
params = {'beta': 0.03063, 'gamma': 0.2}
SIR_model = fg.ModelConfiguration(SIR_schema).add_parameter(**params).get_networks(contact_network=G_csr)
# --- Initial condition (random) ---
ic = {'percentage': {'S': 99, 'I': 1, 'R': 0}}
# --- Run simulation ---
sim = fg.Simulation(SIR_model, initial_condition=ic, stop_condition={'time': 120}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-12.png'))
time, state_count, *_ = sim.get_results()
data = {'time': time}
for i, comp in enumerate(SIR_schema.compartments):
    data[comp] = state_count[i, :]
df = pd.DataFrame(data)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-12.csv'), index=False)

import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
import pandas as pd

# Load the network
network_path = os.path.join('output', 'network.npz')
G_csr = sparse.load_npz(network_path)

# Get degree info from CSR matrix
degrees = np.array(G_csr.sum(axis=1)).flatten()
hub_indices = np.argsort(-degrees)[:3]
N = G_csr.shape[0]
X0 = np.zeros(N, dtype=int)
X0[hub_indices] = 1  # Infect top 3 hubs
init_cond_2 = {'exact': X0}

# SIR ModelSchema (as previously defined)
SIR_model_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta'
    )
)

params = {'beta': 0.004179573493203892, 'gamma': 0.04}
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(**params)
    .get_networks(contact_network_layer=G_csr)
)

sim = fg.Simulation(
    SIR_instance,
    initial_condition=init_cond_2,
    stop_condition={'time': 120},
    nsim=5
)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-12.png'))

# Export results
results_path = os.path.join('output','results-12.csv')
time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i, label in enumerate(['S','I','R']):
    simulation_results[label] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(results_path, index=False)

results_paths = {'csv': results_path,'png': os.path.join('output','results-12.png')}
results_paths