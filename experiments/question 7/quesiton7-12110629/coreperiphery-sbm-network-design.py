
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
# Parameters (tunable, but here use representative mid-range values in stated ranges)
N = 500  # total nodes (banks)
core_fraction = 0.2
n_core = int(N * core_fraction)
n_periphery = N - n_core
# group labels
sizes = [n_core, n_periphery]
groups = ['core']*n_core + ['periphery']*n_periphery
# Block connection probabilities (representative medians in intervals)
p_cc = 0.88  # core-core
p_cp = 0.25  # core-periphery
p_pp = 0.025 # periphery-periphery
probs = [[p_cc, p_cp], [p_cp, p_pp]]
# Generate SBM (stochastic block model)
G = nx.stochastic_block_model(sizes, probs, seed=42)
# Add group labels as node attributes
for i, grp in enumerate(groups):
    G.nodes[i]['group'] = grp
# Save adjacency matrix as sparse file
adj = nx.to_scipy_sparse_array(G, format='csr')
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
net_path = os.path.join(output_dir, 'coreperiphery-sbm-network.npz')
sparse.save_npz(net_path, adj)
# Also save node group membership as auxiliary metadata for simulation
node_grp_path = os.path.join(output_dir, 'nodegroups-coreperiphery.txt')
with open(node_grp_path, 'w') as f:
    for i, grp in enumerate(groups):
        f.write(f"{i},{grp}\n")
# Empirical degree moments & group connections
k_all = np.array([d for n,d in G.degree()])
k_mean = k_all.mean()
k2_mean = (k_all**2).mean()
# Size of largest connected component (normalized)
components = [len(c) for c in nx.connected_components(G)]
gcc_size = max(components)
gcc_frac = gcc_size / N
# Degree distribution plot
plt.figure(figsize=(6, 4))
plt.hist(k_all, bins=30, alpha=0.7, color='b')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution (Core-Periphery SBM)')
deg_dist_path = os.path.join(output_dir, 'degree-dist-coreperiphery-sbm.png')
plt.tight_layout()
plt.savefig(deg_dist_path)
plt.close()
# Clustering coefficient (global, local median)
glob_clust = nx.transitivity(G)
median_clust = np.median(list(nx.clustering(G).values()))
# Assortativity by degree, by group
assort_deg = nx.degree_assortativity_coefficient(G)
group_labels = [G.nodes[i]['group'] for i in G.nodes]
assort_grp = nx.attribute_assortativity_coefficient(G, 'group')
# Visualization (for N≤250: show full network with group color); for N=500, skip as too dense
# Save metrics
diagnostics = {
    'mean_degree': float(k_mean),
    'second_degree_moment': float(k2_mean),
    'gcc_fraction': float(gcc_frac),
    'global_clustering': float(glob_clust),
    'median_local_clustering': float(median_clust),
    'degree_assortativity': float(assort_deg),
    'group_assortativity': float(assort_grp),
    'node_group_path': node_grp_path,
    'degree_dist_plot': deg_dist_path,
    'network_path': net_path
}
