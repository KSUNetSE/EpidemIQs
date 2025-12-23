
# Fix fastgemf initial condition requirements: input must be percentage summing to 100
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import networkx as nx
import os
import pandas as pd

N = 1000
p = 8/(N-1)
G = nx.erdos_renyi_graph(N, p)
sparse.save_npz(os.path.join(os.getcwd(), "output", "network.npz"), nx.to_scipy_sparse_array(G))

degrees = np.array([d for n, d in G.degree()])
k_mean = degrees.mean()
k_sq_mean = (degrees**2).mean()
q = (k_sq_mean-k_mean)/k_mean

gamma = 0.1
R0 = 3
beta = R0 * gamma / q

SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network')
    .add_node_transition('recovery', 'I', 'R', rate='gamma')
    .add_edge_interaction('infection', 'S', 'I', inducer='I', network_layer='contact_network', rate='beta')
)
net_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))
SIR_instance = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact_network=net_csr)
)
# Trick: Give as percentage summing to 100, 1 infected only
init = {'percentage': {'S': 99, 'I': 1, 'R': 0}}  # 990 S, 10 I in population 1000
sim = fg.Simulation(SIR_instance, initial_condition=init, stop_condition={'time': 200}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-11.png'))
time, state_count, *_ = sim.get_results()
data = {'time': time}
for i, c in enumerate(['S', 'I', 'R']):
    data[c] = state_count[i,:]
pd.DataFrame(data).to_csv(os.path.join(os.getcwd(),'output','results-11.csv'), index=False)
S_final = state_count[0, -1]
I_final = state_count[1, -1]
S_final, I_final, k_mean, k_sq_mean, beta