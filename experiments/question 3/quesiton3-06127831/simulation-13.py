
# Step 3 (Optional): Sweep beta for static and temporal network near threshold to confirm epidemic threshold. (i=1, j=3-6)
# For demonstration, run j=3: static at lower beta just below threshold, j=4: temporal at lower beta. N_sim=50 for speed.
import os
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import pandas as pd
from collections import Counter


#####################
# Static j=3, beta=0.008, gamma=1.0 (just below threshold)
dir_output = os.path.join(os.getcwd(), 'output/EpidemiIQs-4.1/Q6/4')
A_path = os.path.join(dir_output, 'static-aggregated-adjacency.npz')
results_csv = os.path.join(dir_output, 'results-13.csv')
results_png = os.path.join(dir_output,  'results-13.png')
N = 1000
beta = 0.008
gamma = 1.0
n_sim = 50
run_time = 365
import random
random.seed(24)
np.random.seed(24)
G_csr = sparse.load_npz(A_path)
# Compute the degree of each node (row-wise sum of non-zeros)
# Use .A1 to convert matrix to a flat NumPy array
degrees = G_csr.data

weight_counts = Counter(degrees)

# Convert to sorted list (by weight)
sorted_weight_counts = sorted(weight_counts.items())
edges_no= G_csr.nnz
for weight, count in sorted_weight_counts:
    
    print(f"Weight: {weight}, Frequency: {count/(edges_no)}")
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
SIR_config = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact=G_csr)
)
X0 = np.zeros(N, dtype=int)
from numpy.random import choice
infected_idxs = choice(N, size=5, replace=False)
X0[infected_idxs] = 1
init_cond = {'exact': X0}
sim = fg.Simulation(SIR_config, initial_condition=init_cond, stop_condition={'time': run_time}, nsim=n_sim)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=results_png)
time, state_count, *_ = sim.get_results()
sim_results = {'time': time}
compartments = SIR_schema.compartments
for i in range(state_count.shape[0]):
    sim_results[compartments[i]] = state_count[i, :]
data = pd.DataFrame(sim_results)
data.to_csv(results_csv, index=False)

net_nodes = G_csr.shape[0]
net_edges = G_csr.nnz
(net_nodes, net_edges)
