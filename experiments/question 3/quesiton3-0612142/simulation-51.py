
# Simulation 5: Static random network (calibrated beta) with IC1 (single seed infectious, others susceptible)
import os
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import fastgemf as fg
import random

# Load network
network_path = os.path.join(os.getcwd(), 'output', 'exp2-static-random.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]

# SIR Model configuration
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery1',
        from_state='I',
        to_state='R',
        rate='mu'
    )
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta'
    )
)

# Parameters: static scenario
params = {'beta': 0.0469, 'mu': 1.0}
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(**params)
    .get_networks(contact_network_layer=G_csr)
)

# Initial Condition IC1: single infectious node, rest susceptible
X0 = np.zeros(N, dtype=int)
single_seed = random.randint(0, N-1)
X0[single_seed] = 1
initial_condition = {'exact': X0}

# Simulation setup
stop_time = 300  # 300 time units expected to cover the outbreak
n_realizations = 100

sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': stop_time}, nsim=n_realizations)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-51.png'))

time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i, state in enumerate(SIR_model_schema.compartments):
    simulation_results[state] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-51.csv'), index=False)

# For returning details
N_nodes = N
N_edges = G_csr.nnz // 2  # undirected
output_csv = os.path.join(os.getcwd(), 'output', 'results-51.csv')
output_png = os.path.join(os.getcwd(), 'output', 'results-51.png')
N_nodes, N_edges, output_csv, output_png