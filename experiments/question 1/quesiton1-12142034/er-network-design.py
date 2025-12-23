
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import os

# Parameters
N = 5000
mean_deg = 8
# ER parameters
p = mean_deg / (N - 1)

# Construct the ER network
G_er = nx.erdos_renyi_graph(n=N, p=p, seed=42)
degrees_er = [d for n, d in G_er.degree()]
k_mean_er = np.mean(degrees_er)
k2_mean_er = np.mean(np.square(degrees_er))

# Save the ER network adjacency matrix
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
er_path = os.path.join(output_dir, "er-network.npz")
sparse.save_npz(er_path, nx.to_scipy_sparse_array(G_er))

# Plot Degree Distribution Histogram
plt.figure(figsize=(8,6))
plt.hist(degrees_er, bins=range(0, max(degrees_er)+1), alpha=0.7, color='b')
plt.title('ER Network Degree Distribution')
plt.xlabel('Degree k')
plt.ylabel('Count')
plt.tight_layout()
er_hist_path = os.path.join(output_dir, "er-degree-histogram.png")
plt.savefig(er_hist_path)
plt.close()

# Largest Connected Component
gcc_nodes_er = max(nx.connected_components(G_er), key=len)
gcc_size_er = len(gcc_nodes_er)
gcc_frac_er = gcc_size_er / N

# Clustering
clustering_er = nx.average_clustering(G_er)

# Assortativity
assortativity_er = nx.degree_pearson_correlation_coefficient(G_er)

return_vars = [
    "er_path", "er_hist_path", "k_mean_er", "k2_mean_er",
    "gcc_size_er", "gcc_frac_er", "clustering_er", "assortativity_er"
]