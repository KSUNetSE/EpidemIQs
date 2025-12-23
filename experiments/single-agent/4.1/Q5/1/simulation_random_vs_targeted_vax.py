
# Chain-of-thought:
# - First, construct a random Poisson network (Erdős-Rényi approximation) with N=5000, mean degree z=3.
# - Simulate an SIR epidemic with R0=4, using calculated beta and gamma for this network.
# - Simulation 1: Apply random vaccination, remove fraction 0.75 of nodes randomly, run SIR simulation, see if outbreak is possible (should stop epidemic).
# - Simulation 2: Remove all nodes with k=10 (about pk10 proportion), run SIR simulation, check if outbreak is possible (should NOT be sufficient to stop epidemic as q_new > 1).
# - For both, store time series and final epidemic size for comparison. Save results and plots.

import numpy as np
import networkx as nx
import fastgemf as fg
import pandas as pd
import scipy.sparse as sparse
import os

np.random.seed(1)

# Network parameters
N = 5000
z = 3
p = z/(N-1)
G = nx.erdos_renyi_graph(N, p)
A_sparse = nx.to_scipy_sparse_array(G)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network.npz'), A_sparse)

# SIR parameters
R0 = 4
gamma = 0.25
q = 4
beta = R0 * gamma / q

# SIR model schema
def make_SIR_schema():
    return (
        fg.ModelSchema("SIR")
        .define_compartment(['S', 'I', 'R'])
        .add_network_layer('contact_network_layer')
        .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
        .add_edge_interaction(
            name='infection', from_state='S', to_state='I', inducer='I',
            network_layer='contact_network_layer', rate='beta')
    )

# Random vaccination - remove 75% nodes
nodes = np.arange(N)
np.random.shuffle(nodes)
vax_frac = 0.75
num_vax = int(vax_frac * N)
vaxed_nodes = set(nodes[:num_vax])
remaining_nodes = np.setdiff1d(nodes, list(vaxed_nodes))
G1 = G.subgraph(remaining_nodes).copy()
A1_sparse = nx.to_scipy_sparse_array(G1)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network-random-vax.npz'), A1_sparse)

# Targeted vaccination - remove all nodes with k = 10
k_target = 10
nodes_k10 = [n for n, d in G.degree() if d == k_target]
G2 = G.copy()
G2.remove_nodes_from(nodes_k10)
A2_sparse = nx.to_scipy_sparse_array(G2)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network-degree10-vax.npz'), A2_sparse)

# SIR model setup
model_schema = make_SIR_schema()
def make_model_config(A_sparse):
    return (
        fg.ModelConfiguration(model_schema)
        .add_parameter(beta=beta, gamma=gamma)
        .get_networks(contact_network_layer=A_sparse)
    )

# Function for initial condition: infect 5 random nodes, rest susceptible

def initial_condition_random(G):
    N = len(G)
    X0 = np.zeros(N, dtype=int)
    infected = np.random.choice(N, 5, replace=False)
    X0[infected] = 1
    return {'exact': X0}

# Run simulation for each scenario and save to distinct files
model1 = make_model_config(A1_sparse)
model2 = make_model_config(A2_sparse)

sim1 = fg.Simulation(model1, initial_condition=initial_condition_random(G1), stop_condition={'time': 100}, nsim=3)
sim1.run()
sim1.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-11.png'))
time, state_count, *_ = sim1.get_results()
df1 = pd.DataFrame({'time': time, 'S': state_count[0], 'I': state_count[1], 'R': state_count[2]})
df1.to_csv(os.path.join(os.getcwd(),'output','results-11.csv'), index=False)

sim2 = fg.Simulation(model2, initial_condition=initial_condition_random(G2), stop_condition={'time': 100}, nsim=3)
sim2.run()
sim2.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-12.png'))
time, state_count, *_ = sim2.get_results()
df2 = pd.DataFrame({'time': time, 'S': state_count[0], 'I': state_count[1], 'R': state_count[2]})
df2.to_csv(os.path.join(os.getcwd(),'output','results-12.csv'), index=False)
