
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import sparse

# Parameters
def network_output_dir():
    path = os.path.join(os.getcwd(), "output")
    if not os.path.exists(path):
        os.makedirs(path)
    return path

N = 763  # population size
network = nx.complete_graph(N)

# Compute diagnostics
mean_degree = np.mean([deg for _, deg in network.degree()])
second_moment_degree = np.mean([deg**2 for _, deg in network.degree()])

# Check largest component size
gcc_size = len(max(nx.connected_components(network), key=len))

# Degree distribution (should be a delta spike at N-1)
degrees = [deg for _, deg in network.degree()]

# Visualize degree distribution (delta spike)
plt.figure()
plt.hist(degrees, bins=N, color='skyblue', edgecolor='k', align='left')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Degree distribution: Complete Graph (K_763)')
plt.xlim([N-3, N+1])  # Only near the actual value
plot_deg_dist_path = os.path.join(network_output_dir(), "degree-distribution-complete.png")
plt.savefig(plot_deg_dist_path)
plt.close()

# For large N, visualizing the full network is not helpful; skip or use a small sample
# Save the network
net_path = os.path.join(network_output_dir(), "network-complete.npz")
sparse.save_npz(net_path, nx.to_scipy_sparse_array(network))

# Save diagnostics for reporting
stats = {
    'mean_degree': mean_degree,
    'second_degree_moment': second_moment_degree,
    'largest_connected_component': gcc_size,
    'degree_sample': degrees[:10],  # First 10 for proof
    'network_path': net_path,
    'degree_dist_plot_path': plot_deg_dist_path
}

stats
