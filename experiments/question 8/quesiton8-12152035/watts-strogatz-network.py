
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import sparse
import seaborn as sns

# 1. Parameters
N = 10000  # nodes
k = 8      # mean degree
p = 0.1    # rewiring
seed = 42
np.random.seed(seed)

# 2. Create Watts-Strogatz network
G = nx.watts_strogatz_graph(N, k, p, seed=seed)

# 3. Check connectivity
# If not connected, connect largest component, but typically is connected for these params
if not nx.is_connected(G):
    GCC = max(nx.connected_components(G), key=len)
    G = G.subgraph(GCC).copy()

# 4. Calculate centralities and diagnostics
# Degree stats
degrees = np.array([d for n, d in G.degree()])
mean_k = degrees.mean()
second_moment_k = (degrees ** 2).mean()

# Global clustering
clustering_coeff = nx.average_clustering(G)

# Mean shortest path length (May take time, but feasible for this size)
mean_shortest_path = nx.average_shortest_path_length(G)

# Size GCC
gcc_size = len(max(nx.connected_components(G), key=len))

# 5. Plots
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

# Degree distribution
plt.figure(figsize=(8, 4))
sns.histplot(degrees, bins=range(degrees.min(), degrees.max() + 2), kde=False, stat="probability")
plt.title("Degree Distribution (Watts-Strogatz, N=10000, k=8, p=0.1)")
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.tight_layout()
deg_path = os.path.join(output_dir, "ws-degree-dist.png")
plt.savefig(deg_path)
plt.close()

# Clustering and Path length bar
plt.figure(figsize=(6, 4))
plt.bar(["Clustering", "Avg Path Length"], [clustering_coeff, mean_shortest_path], color=["#1f77b4", "#ff7f0e"])
plt.ylabel("Value")
plt.title("Small-World Structure Diagnostics")
bar_path = os.path.join(output_dir, "ws-clustering-path.png")
plt.tight_layout()
plt.savefig(bar_path)
plt.close()

# 6. Save adjacency
net_path = os.path.join(output_dir, "watts-strogatz-graph.npz")
sparse.save_npz(net_path, nx.to_scipy_sparse_array(G))

# Return all diagnostics and paths
res = {
    "mean_k": mean_k,
    "second_moment_k": second_moment_k,
    "clustering": clustering_coeff,
    "mean_shortest_path": mean_shortest_path,
    "gcc_size": gcc_size,
    "network_path": net_path,
    "degree_plot": deg_path,
    "bar_plot": bar_path
}
res