
# Chain-of-Thought: Modeling step for user query comparing SIR spread on a temporal activity-driven network versus its time-aggregated static network.
# 1. We need to generate two versions of the network given N=1000, alpha=0.1, m=2: (A) a temporal activity-driven network (returning a sequence of contact lists or adjacency matrices over T steps); and (B) time-aggregated static network (adjacency weighted by edge frequency over T, for same random seed). Also measure degree statistics.
# 2. Parameters: For R0=3, SIR model, and given static mean degree, we will compute infection and recovery rates for both scenarios (beta/gamma) for use in next modeling step.
# This script creates both types of networks and reports mean degree, <k^2>, and aggregates edge weights.

import numpy as np
import networkx as nx
from scipy import sparse
import os

np.random.seed(42)
N = 1000
alpha = 0.1  # node activation probability per time step
m = 2        # connections per activation
T = 1000     # number of time steps for aggregation window

# (A) Generate temporal network (sequence of edge lists at each time step)
temporal_edges = []  # Each entry is list of (i, j) edges at time t
all_edges = []       # For construction of aggregation
for t in range(T):
    activated = np.where(np.random.rand(N) < alpha)[0]
    edges_t = []
    for i in activated:
        partners = np.random.choice([x for x in range(N) if x != i], m, replace=False)
        for j in partners:
            edge = tuple(sorted([i, j]))
            edges_t.append(edge)
            all_edges.append(edge)
    temporal_edges.append(edges_t)
# (B) Time-aggregated static network
from collections import Counter
edge_counts = Counter(all_edges)  # Frequency for each unique edge
G_agg = nx.Graph()
for (i, j), w in edge_counts.items():
    G_agg.add_edge(i, j, weight=w)
# Save the static network
agg_net_mat = nx.to_scipy_sparse_array(G_agg)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network_static.npz'), agg_net_mat)
# Also compute mean degree and <k^2> for aggregation
k_list = np.array([d for n, d in G_agg.degree()])
mean_k = np.mean(k_list)
mean_k2 = np.mean(k_list ** 2)
# Degree for T=1 (original temporal model instantaneous graph)
# This network is much sparser, instantaneously
G_inst = nx.Graph()
G_inst.add_edges_from(temporal_edges[0])
k_inst_list = np.array([d for n, d in G_inst.degree()])
mean_k_inst = np.mean(k_inst_list)
mean_k2_inst = np.mean(k_inst_list ** 2)
# Degree for the time-averaged temporal network (mean number of edges per time step)
deg_per_snap = [2*len(edges)/N for edges in temporal_edges]
mean_deg_per_snap = np.mean(deg_per_snap)
dict_results = {
    'mean_k_static': float(mean_k),
    'mean_k2_static': float(mean_k2),
    'mean_k_temporal_instant': float(mean_k_inst),
    'mean_k2_temporal_instant': float(mean_k2_inst),
    'mean_deg_per_timestep': float(mean_deg_per_snap),
    'agg_network_path': os.path.join(os.getcwd(), 'output', 'network_static.npz'),
}
dict_results
