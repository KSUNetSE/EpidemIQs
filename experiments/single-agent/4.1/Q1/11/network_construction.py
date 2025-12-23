
# PLAN (Chain of Thought):
# 1. We need to compare SEIR epidemic dynamics on a homogeneous (Erdős–Rényi, ER) and a degree-heterogeneous network (Barabási–Albert, BA/scale-free).
# 2. For both, we will construct static networks of the same size and similar mean degree.
# 3. Save both networks for use in simulation: network_homogeneous.npz and network_heterogeneous.npz.
# 4. Compute <k> and <k^2> for both networks for later analytical threshold calculation.
# 5. Plot and save degree distributions for both networks.
# 6. Save code as network_construction.py

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import sparse

output_folder = os.path.join(os.getcwd(), 'output')
os.makedirs(output_folder, exist_ok=True)

N = 1000  # Total population size
mean_degree = 8  # Try to keep mean degree similar
# 1. Homogeneous (ER) Network
p = mean_degree / (N - 1)  # ER G(N, p): expected mean degree ~ p*(N-1) ~ mean_degree
G_er = nx.erdos_renyi_graph(N, p, seed=42)

# 2. Degree-heterogeneous (BA) Scale-Free Network
m = mean_degree // 2  # Each new node connects to m existing nodes (approx <k>=2m)
G_ba = nx.barabasi_albert_graph(N, m, seed=42)

# Save networks
sparse.save_npz(os.path.join(output_folder, "network_homogeneous.npz"), nx.to_scipy_sparse_array(G_er))
sparse.save_npz(os.path.join(output_folder, "network_heterogeneous.npz"), nx.to_scipy_sparse_array(G_ba))

# Compute degree stats for both
degrees_er = np.array([d for n, d in G_er.degree()])
degrees_ba = np.array([d for n, d in G_ba.degree()])
k_er_mean = degrees_er.mean()
k_er_2nd_moment = (degrees_er**2).mean()
k_ba_mean = degrees_ba.mean()
k_ba_2nd_moment = (degrees_ba**2).mean()

# Plot degree distributions
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(degrees_er, bins=20, color='skyblue', edgecolor='black')
plt.title('ER (Homogeneous) Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.subplot(1,2,2)
plt.hist(degrees_ba, bins=20, color='salmon', edgecolor='black')
plt.title('BA (Heterogeneous) Degree Distribution')
plt.xlabel('Degree')
plt.savefig(os.path.join(output_folder, 'degree_distributions.png'))
plt.close()

# Return statistics for documentation
network_stats = {
    'k_er_mean': k_er_mean,
    'k_er_2nd_moment': k_er_2nd_moment,
    'k_ba_mean': k_ba_mean,
    'k_ba_2nd_moment': k_ba_2nd_moment
}
