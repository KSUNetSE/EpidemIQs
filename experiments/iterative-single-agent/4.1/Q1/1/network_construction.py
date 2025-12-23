
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