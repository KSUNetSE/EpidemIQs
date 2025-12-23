
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import os

# Set parameters for the population and network
N = 500  # Network size
p = 0.05 # Edge probability for ER network, to get reasonable <k>
np.random.seed(42)  # reproducibility

# Create ER random graph
G = nx.erdos_renyi_graph(N, p)

# Save the network in sparse format
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
sparse.save_npz(network_path, nx.to_scipy_sparse_array(G))

# Compute and display <k> and <k^2>
degrees = np.array([deg for n, deg in G.degree()])
mean_k = degrees.mean()
mean_k2 = np.mean(degrees ** 2)

# Plot degree distribution
plt.figure(figsize=(7, 4))
plt.hist(degrees, bins=range(degrees.min(), degrees.max() + 2), color='steelblue', alpha=0.8)
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.title('Degree Distribution (Erdős-Rényi, N=500, p=0.05)')
plt.grid(True)
plot_path = os.path.join(os.getcwd(), 'output', 'network_degree_dist.png')
plt.savefig(plot_path)
plt.close()

# Return key results for modeling phase
(mean_k, mean_k2, network_path, plot_path)