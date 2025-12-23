
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import os

# Set parameters
N = 1000  # population size
ER_p = 0.01  # ER probability for homogeneous mixing network
BA_m = 3     # Barabasi-Albert parameter for heterogeneous network

# 1. Homogeneous (Erdos-Renyi) network
G_hom = nx.erdos_renyi_graph(N, ER_p)
mean_deg_hom = np.mean([d for _, d in G_hom.degree()])
k2_hom = np.mean([d ** 2 for _, d in G_hom.degree()])
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network_hom.npz'), nx.to_scipy_sparse_array(G_hom))

# 2. Heterogeneous (Barabasi-Albert) network
G_het = nx.barabasi_albert_graph(N, BA_m)
mean_deg_het = np.mean([d for _, d in G_het.degree()])
k2_het = np.mean([d ** 2 for _, d in G_het.degree()])
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network_het.npz'), nx.to_scipy_sparse_array(G_het))

# Make Degree Distribution Plots
plt.figure(figsize=(10,5))
deghist_hom = [d for n, d in G_hom.degree()]
deghist_het = [d for n, d in G_het.degree()]
plt.hist(deghist_hom, bins=30, alpha=0.5, label='ER Network (Homogeneous)')
plt.hist(deghist_het, bins=30, alpha=0.5, label='BA Network (Heterogeneous)')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.title('Degree Distributions (Homogeneous vs. Heterogeneous Network)')
plt.legend()
plt.savefig(os.path.join(os.getcwd(), 'output', 'degree_dist.png'))
plt.close()

# Return stats for further use
(mean_deg_hom, k2_hom, mean_deg_het, k2_het)
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import sparse

# Parameters for network construction based on standard SIR literature
# Let's make a static network with N=1000 nodes and mean degree <k>=8 (moderately connected, useful for SIR spread test)
N = 1000
mean_k = 8
p = mean_k / (N-1)

# Generate Erdős-Rényi random graph (ER)
G = nx.erdos_renyi_graph(N, p)

# Save the network
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, "network.npz")
sparse.save_npz(network_path, nx.to_scipy_sparse_array(G))

# Degree metrics
degrees = np.array([d for n, d in G.degree()])
mean_degree = degrees.mean()
second_moment = (degrees**2).mean()

# Degree histogram for appendices
plt.figure()
plt.hist(degrees, bins=30, edgecolor='black')
plt.title('Degree Distribution of ER Network')
plt.xlabel('Degree')
plt.ylabel('Count')
degree_hist_path = os.path.join(output_dir, 'degree_dist.png')
plt.savefig(degree_hist_path)
plt.close()

# Return info for following phase
(network_path, mean_degree, second_moment, degree_hist_path)