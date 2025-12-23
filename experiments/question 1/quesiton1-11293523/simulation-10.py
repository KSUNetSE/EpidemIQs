
# Scenario (i=1, j=0): Heterogeneous (scale-free), random seeding, SEIR
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import os

N = 1000
beta = 0.00871717606543263
sigma = 0.2
gamma = 0.1
network_path = '/Users/hosseinsamaei/phd/gemf_llm/output/scaleffreenetwork.npz'
current_directory = os.getcwd()
results_csv = os.path.join(current_directory, 'output', 'results-10.csv')
results_png = os.path.join(current_directory, 'output', 'results-10.png')
code_path = os.path.join(current_directory, 'output', 'simulation-10.py')

if not os.path.exists(network_path):
    # If scale-free network is missing, generate and save it (see justification below).
    import networkx as nx
    deg_seq = np.random.zipf(2, N)
    while sum(deg_seq) % 2 != 0:
        deg_seq[0] += 1
    G = nx.configuration_model(deg_seq)
    G = nx.Graph(G)  # remove multiedges and selfloops
    G.remove_edges_from(nx.selfloop_edges(G))
    G_csr = sparse.csr_matrix(nx.to_scipy_sparse_array(G))
    sparse.save_npz(network_path, G_csr)
else:
    G_csr = sparse.load_npz(network_path)

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

np.random.seed(42)
initial_X = np.zeros(N, dtype=int)
infectious_seeds = np.random.choice(N, 5, replace=False)
initial_X[infectious_seeds] = 2
initial_condition = {'exact': initial_X}

sr = 300
sim = fg.Simulation(SEIR_instance, initial_condition=initial_condition, stop_condition={'time': 200}, nsim=sr)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=results_png)
time, state_count, *_ = sim.get_results()
results = {'time': time}
for idx, comp in enumerate(SEIR_schema.compartments):
    results[f'{comp}'] = state_count[idx, :]
data = pd.DataFrame(results)
data.to_csv(results_csv, index=False)
num_nodes = G_csr.shape[0]
num_edges = int(G_csr.sum() // 2)
network_details = {'nodes': int(num_nodes), 'edges': int(num_edges)}

output_paths = {results_csv: 'SEIR Simulated Trajectories (scale-free, random seed)', results_png: 'SEIR Dynamics Plot (scale-free, random seed)'}
simulated_model_details = {'model': 'SEIR', 'network_type': 'scale_free', 'N': N, 'params': {'beta': beta, 'sigma': sigma, 'gamma': gamma}, 'initial_condition': '5 random infectious, rest susceptible', 'num_realizations': sr, 'network_details': network_details}
plot_path = {results_png: 'SEIR Dynamics Plot (scale-free, random seeding)'}
stored_result_path = output_paths
(simulated_model_details, stored_result_path, plot_path, network_details)
