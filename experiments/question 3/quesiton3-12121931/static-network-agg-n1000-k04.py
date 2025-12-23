
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import sparse

def ensure_output_dir():
    directory = os.path.join(os.getcwd(), "output")
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# 1a. Build the time-aggregated static network with N=1000, mean degree 0.4, edges weighted by frequency.
N = 1000
mean_k = 0.4
n_edges = int((N * mean_k) // 2)

# Generate edge list for random undirected graph
edges = set()
while len(edges) < n_edges:
    i, j = np.random.choice(N, 2, replace=False)
    if i > j:
        i, j = j, i
    if i != j:
        edges.add((i, j))
edge_list = list(edges)

# Assign edge weights: for a matching mean weight, assign all 1 (since total contacts per period = mean degree)
# This matches having a mean number of contacts per infectious period as specified
weights = np.ones(len(edge_list))

G_static = nx.Graph()
G_static.add_nodes_from(range(N))
for edge, w in zip(edge_list, weights):
    G_static.add_edge(edge[0], edge[1], weight=w)

# Calculations for centralities/metrics
static_degrees = np.array([deg for n, deg in G_static.degree()])
static_mean_k = static_degrees.mean()
static_k2 = (static_degrees ** 2).mean()

# Plots: degree distribution and weight distribution
output_dir = ensure_output_dir()
plt.figure(figsize=(6, 4))
plt.hist(static_degrees, bins=np.arange(static_degrees.max()+2)-0.5, alpha=0.7)
plt.xlabel("Degree (k)")
plt.ylabel("Count")
plt.title("Degree Distribution (Static Aggregated Network)")
plt.savefig(os.path.join(output_dir, "static-degree-histogram.png"))
plt.close()

# Since weights are all 1 here, weight distribution is trivial

# Save the network for simulation phase
static_network_path = os.path.join(output_dir, "network-static-n1000-k04.npz")
sparse.save_npz(static_network_path, nx.to_scipy_sparse_array(G_static, weight='weight'))

# Perform giant connected component check
gcc = max(nx.connected_components(G_static), key=len)
gcc_size = len(gcc)

return_vars = [
    'static_network_path',
    'static_mean_k',
    'static_k2',
    'gcc_size',
    'output_dir'
]