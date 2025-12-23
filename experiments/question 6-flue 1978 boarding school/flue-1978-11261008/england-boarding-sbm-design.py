

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sparse

# 1. Parameters for English boarding school SBM
N = 763
n_blocks = 4
block_sizes = [191, 191, 191, 190]  # sum = 763
total = sum(block_sizes)
assert total == N, 'Block sizes must sum to N'

# Connection probabilities (within and between blocks)
p_within = 0.09
p_between = 0.01
prob_matrix = [[p_within if i == j else p_between for j in range(n_blocks)] for i in range(n_blocks)]

# 2. Generate SBM
g = nx.stochastic_block_model(block_sizes, prob_matrix, seed=42)
# Remove self-loops if present
g.remove_edges_from(nx.selfloop_edges(g))

# 3. Compute degree statistics & structure
k_arr = np.array([deg for node, deg in g.degree()])
k_mean = k_arr.mean()
k2_mean = (k_arr**2).mean()

# Optionally check connected components
connected_components = list(nx.connected_components(g))
largest_cc = max(connected_components, key=len)
size_largest_cc = len(largest_cc)

# Local clustering (transitivity)
gcc = nx.transitivity(g)

# 4. Save the network
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, "england-boarding-sbm.npz")
sparse.save_npz(network_path, nx.to_scipy_sparse_array(g))

# 5. Plot degree histogram
plt.figure(figsize=(7,5))
plt.hist(k_arr, bins=range(int(k_arr.min()), int(k_arr.max())+2), color='skyblue', edgecolor='black', alpha=0.75)
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('SBM Boarding School Network (763 nodes) Degree Distribution')
plt.tight_layout()
degree_hist_path = os.path.join(output_dir, 'england-boarding-degree-histogram.png')
plt.savefig(degree_hist_path)
plt.close()

# 6. Visualize network blocks (spring layout, color by block; show only 200 nodes for speed)
plt.figure(figsize=(7, 7))
# For readability, plot subgraph of first 200 nodes
sub_nodes = list(range(200))
sg = g.subgraph(sub_nodes)
block_labels = []
running = 0
for b, s in enumerate(block_sizes):
    block_labels += [b]*s
block_colors = [block_labels[node] for node in sub_nodes]
pos = nx.spring_layout(sg, seed=42)
nx.draw_networkx_nodes(sg, pos, node_color=block_colors, cmap=plt.get_cmap('tab10'), node_size=30)
nx.draw_networkx_edges(sg, pos, alpha=0.35)
plt.title('SBM Network Sample (First 200 Nodes) - Colored by Dorm Block')
plt.axis('off')
block_plot_path = os.path.join(output_dir, 'england-boarding-blocks-layout.png')
plt.savefig(block_plot_path)
plt.close()

# 7. Save the centrality/diagnostic results for reporting
network_metrics = {
    'n_nodes': g.number_of_nodes(),
    'n_edges': g.number_of_edges(),
    'mean_degree': float(np.round(k_mean,2)),
    'second_degree_moment': float(np.round(k2_mean,2)),
    'largest_cc_size': int(size_largest_cc),
    'global_clustering_coefficient': float(np.round(gcc,4)),
    'degree_dist_min': int(k_arr.min()),
    'degree_dist_max': int(k_arr.max()),
    'block_sizes': block_sizes
}

# Return paths and metrics
result = {
    'network_path': network_path,
    'degree_hist_path': degree_hist_path,
    'block_plot_path': block_plot_path,
    'network_metrics': network_metrics
}
