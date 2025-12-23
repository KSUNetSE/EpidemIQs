
# Step 1: Static Aggregated SIR Simulation (Scenario 1, i=1, j=1)
# Plan: Load the static network, setup SIR with FastGEMF at analytic R0=3 (beta=0.0166, gamma=1.0), seed 5 I randomly, run n_sim realizations, save results and plot.
import os
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import pandas as pd

from numpy.random import choice, seed
import matplotlib.pyplot as plt

# Parameters and paths
dir_output = os.path.join(os.getcwd(), 'output')
A_path = os.path.join(dir_output, 'static-aggregated-adjacency.npz')
results_csv = os.path.join(dir_output, 'results-11.csv')
results_png = os.path.join(dir_output, 'results-11.png')
N = 1000
beta = 0.0166
gamma = 1.0
n_sim = 100  # 100 realizations (collapsing statistical error)
run_time = 365  # Long enough to finish outbreak

# --- Load static aggregated network ---
G_csr = sparse.load_npz(A_path)
assert G_csr.shape[0] == N

# -- Setup Model Schema (SIR)
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact', rate='beta')
    .add_node_transition(
        name='recovery', from_state='I', to_state='R', rate='gamma')
)

# -- Setup ModelConfig with network and params
SIR_config = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact=G_csr)
)

# -- Seed initial infection, exactly 5 random I, rest S
seed(42)
X0 = np.zeros(N, dtype=int)  # S=0, I=1, R=2
infected_idxs = choice(N, size=5, replace=False)
X0[infected_idxs] = 1
init_cond = {'exact': X0}

# -- Run simulation, multiple stochastic realizations
sim = fg.Simulation(SIR_config, initial_condition=init_cond, stop_condition={'time': run_time}, nsim=n_sim)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=results_png)
# Process and save to CSV

time, state_count, *_ = sim.get_results()
sim_results = {'time': time}
compartments = SIR_schema.compartments
for i in range(state_count.shape[0]):
    sim_results[compartments[i]] = state_count[i, :]
data = pd.DataFrame(sim_results)
data.to_csv(results_csv, index=False)
# Provide number of nodes, edges for report
net_edges = G_csr.nnz
net_nodes = G_csr.shape[0]
(net_nodes, net_edges)
