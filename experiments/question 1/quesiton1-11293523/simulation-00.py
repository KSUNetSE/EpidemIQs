
# Scenario (i=0, j=0): Homogeneous-mixing (complete graph), random seeding, SEIR
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import os
import random

# 1. Parameters
N = 1000
beta = 0.25
sigma = 0.2
gamma = 0.1
network_path = '/Users/hosseinsamaei/phd/gemf_llm/output/completegraphnetwork.npz'
current_directory = os.getcwd()
results_csv = os.path.join(current_directory, 'output', 'results-00.csv')
results_png = os.path.join(current_directory, 'output', 'results-00.png')
code_path = os.path.join(current_directory, 'output', 'simulation-00.py')

# 2. Load Complete Graph CSR Matrix
if not os.path.exists(network_path):
    # Create complete graph if file missing
    import networkx as nx
    G = nx.complete_graph(N)
    G_csr = sparse.csr_matrix(nx.to_scipy_sparse_array(G))
    sparse.save_npz(network_path, G_csr)
else:
    G_csr = sparse.load_npz(network_path)

# 3. Build SEIR Model Schema
SEIR_schema = (
    fg.ModelSchema('SEIR')
    .define_compartment(['S', 'E', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_edge_interaction(name='infection', from_state='S', to_state='E', inducer='I', network_layer='contact_network_layer', rate='beta')
    .add_node_transition(name='incubation', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
)

SEIR_instance = (
    fg.ModelConfiguration(SEIR_schema)
    .add_parameter(beta=beta, sigma=sigma, gamma=gamma)
    .get_networks(contact_network_layer=G_csr)
)
print(SEIR_instance)

# 4. Initial condition: 995 susceptible, 5 infectious (random)
np.random.seed(42)
initial_X = np.zeros(N, dtype=int) # all S (state 0)
infectious_seeds = np.random.choice(N, 5, replace=False)
initial_X[infectious_seeds] = 2 # 'I' is index 2 in ['S','E','I','R']
initial_condition = {'exact': initial_X}

# 5. Setup and run simulation
sr = 300 # Number of stochastic realizations (good tradeoff accuracy vs time)
sim = fg.Simulation(SEIR_instance, initial_condition=initial_condition, stop_condition={'time': 200}, nsim=sr)
sim.run()

# 6. Plot and save results
sim.plot_results(show_figure=False, save_figure=True, save_path=results_png)
time, state_count, *_ = sim.get_results() # shape: (num_compartments, num_timepoints)
results = {'time': time}
for idx, comp in enumerate(SEIR_schema.compartments):
    results[f'{comp}'] = state_count[idx, :]
data = pd.DataFrame(results)
data.to_csv(results_csv, index=False)

# 7. Network details
num_nodes = G_csr.shape[0]
num_edges = (G_csr.sum() // 2) # Undirected complete graph
network_details = {'nodes': int(num_nodes), 'edges': int(num_edges)}

output_paths = {results_csv: 'SEIR Simulated Trajectories (complete graph, random seed)', results_png: 'SEIR Dynamics Plot (complete graph, random seed)'}

simulated_model_details = {'model': 'SEIR', 'network_type': 'complete_graph', 'N': N, 'params': {'beta': beta, 'sigma': sigma, 'gamma': gamma}, 'initial_condition': '5 random infectious, rest susceptible', 'num_realizations': sr, 'network_details': network_details}

stored_result_path = output_paths
plot_path = {results_png: 'SEIR Dynamics Plot (complete graph, random seeding)'}

# Return for evaluation
(simulated_model_details, stored_result_path, plot_path, network_details)
