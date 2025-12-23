
import networkx as nx
import numpy as np
import os
from scipy import sparse
import matplotlib.pyplot as plt

# Set parameters
N = 1000  # Number of nodes
k = 10    # Mean degree
p = 0.05  # Rewiring probability

# Step 3: Constructing the Watts-Strogatz small-world network
g = nx.watts_strogatz_graph(n=N, k=k, p=p)

# Step 4: Compute network metrics
# Mean degree <k>
degrees = np.array([d for n, d in g.degree()])
mean_k = degrees.mean()
# Second moment <k^2>
mean_k2 = np.mean(degrees**2)
# Global clustering coefficient
clustering = nx.average_clustering(g)
# Average shortest path length (on GCC)
gcc_nodes = max(nx.connected_components(g), key=len)
gcc_subgraph = g.subgraph(gcc_nodes)
avg_path_length = nx.average_shortest_path_length(gcc_subgraph)
# Save metrics for reporting
metrics = {
    'mean_degree': mean_k,
    'second_degree_moment': mean_k2,
    'global_clustering': clustering,
    'gcc_size': len(gcc_nodes),
    'average_shortest_path': avg_path_length
}

# Step 5: Save adjacency matrix (as a sparse format)
if not os.path.exists(os.path.join(os.getcwd(), "output")):
    os.makedirs(os.path.join(os.getcwd(), "output"))
adj_sp = nx.to_scipy_sparse_array(g)
network_path = os.path.join(os.getcwd(), "output", "ws-smallworld-1000-nodes.npz")
sparse.save_npz(network_path, adj_sp)

# Step 6: Diagnostic visualizations
# Degree histogram
plt.figure(figsize=(8,5))
plt.hist(degrees, bins=np.arange(degrees.min(), degrees.max()+2)-0.5, color='dodgerblue', edgecolor='black')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Degree distribution (Watts–Strogatz, N=1000, k=10, p=0.05)')
deg_hist_path = os.path.join(os.getcwd(), "output", "degree-distribution-ws-1000-nodes.png")
plt.tight_layout()
plt.savefig(deg_hist_path)
plt.close()

# Clustering coefficient distribution
clus_vals = list(nx.clustering(g).values())
plt.figure(figsize=(8,5))
plt.hist(clus_vals, bins=30, color='orange', edgecolor='black')
plt.xlabel('Clustering coefficient')
plt.ylabel('Number of nodes')
plt.title('Clustering coefficient distribution (WS, N=1000, p=0.05)')
clus_hist_path = os.path.join(os.getcwd(), "output", "clustering-distribution-ws-1000-nodes.png")
plt.tight_layout()
plt.savefig(clus_hist_path)
plt.close()

# Return relevant data
return_vars = [
    'metrics',
    'network_path',
    'deg_hist_path',
    'clus_hist_path',
    'degrees',
    'clus_vals'
]
