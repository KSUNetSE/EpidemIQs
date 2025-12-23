
import os, networkx as nx, scipy.sparse as sparse, numpy as np, matplotlib.pyplot as plt
# Parameters
N = 10000
m0 = 3  # BA network parameter
G = nx.barabasi_albert_graph(N, m0, seed=42)
# Compute degree metrics
degrees = np.array([d for n, d in G.degree()])
mean_k = degrees.mean()
mean_k2 = (degrees**2).mean()
# Save network
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, 'network_static.npz')
sparse.save_npz(network_path, nx.to_scipy_sparse_array(G))
# Plot degree distribution for reference
plt.figure()
plt.hist(degrees, bins=50, log=True)
plt.xlabel('Degree')
plt.ylabel('Frequency (log)')
plt.title('BA Network Degree Distribution')
plot_path = os.path.join(output_dir, 'degree_dist_static.png')
plt.savefig(plot_path)
plt.close()

mean_k, mean_k2, network_path, plot_path