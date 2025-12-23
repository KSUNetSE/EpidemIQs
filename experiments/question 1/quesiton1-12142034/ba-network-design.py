
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import os

# Parameters
N = 5000
mean_deg = 8
m = mean_deg // 2  # BA preferably integer m, so m=4

G_ba = nx.barabasi_albert_graph(n=N, m=m, seed=42)
degrees_ba = [d for n, d in G_ba.degree()]
k_mean_ba = np.mean(degrees_ba)
k2_mean_ba = np.mean(np.square(degrees_ba))

# Save the BA network adjacency matrix
output_dir = os.path.join(os.getcwd(), "output")
ba_path = os.path.join(output_dir, "ba-network.npz")
sparse.save_npz(ba_path, nx.to_scipy_sparse_array(G_ba))

# Plot Degree Distribution Histogram (log-log)
plt.figure(figsize=(8,6))
bins = np.logspace(np.log10(1), np.log10(max(degrees_ba)+1), 30)
plt.hist(degrees_ba, bins=bins, alpha=0.7, color='g')
plt.xscale('log')
plt.yscale('log')
plt.title('BA Network Degree Distribution (log-log)')
plt.xlabel('Degree k (log)')
plt.ylabel('Count (log)')
plt.tight_layout()
ba_hist_log_path = os.path.join(output_dir, "ba-degree-histogram-loglog.png")
plt.savefig(ba_hist_log_path)
plt.close()

# Largest Connected Component
gcc_nodes_ba = max(nx.connected_components(G_ba), key=len)
gcc_size_ba = len(gcc_nodes_ba)
gcc_frac_ba = gcc_size_ba / N

# Clustering
clustering_ba = nx.average_clustering(G_ba)

# Assortativity
assortativity_ba = nx.degree_pearson_correlation_coefficient(G_ba)

return_vars = [
    "ba_path", "ba_hist_log_path", "k_mean_ba", "k2_mean_ba",
    "gcc_size_ba", "gcc_frac_ba", "clustering_ba", "assortativity_ba"
]