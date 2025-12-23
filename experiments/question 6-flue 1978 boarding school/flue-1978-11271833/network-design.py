
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import sparse

# Corrected block sizes for SBM
block_sizes = [113] + [65]*10  # 1 junior house, 10 senior houses
assert sum(block_sizes) == 763, f"Block sum is {sum(block_sizes)} not 763!"

# Edge probabilities
p_in = 0.28
p_out = 0.015
num_blocks = len(block_sizes)
p_matrix = np.full((num_blocks, num_blocks), p_out)
np.fill_diagonal(p_matrix, p_in)

# Build SBM
G = nx.stochastic_block_model(block_sizes, p_matrix, seed=42)

# Assign house labels
node_house = {}
node = 0
for house_id, size in enumerate(block_sizes):
    for _ in range(size):
        node_house[node] = house_id
        node += 1
nx.set_node_attributes(G, node_house, 'house')

# Diagnostics
n = G.number_of_nodes()
deg_seq = [d for _, d in G.degree()]
mean_deg = float(np.mean(deg_seq))
mean_sq_deg = float(np.mean(np.square(deg_seq)))
clustering = float(nx.average_clustering(G))
assort = float(nx.attribute_assortativity_coefficient(G, 'house'))
gcc_size = len(max(nx.connected_components(G), key=len))

# Save network
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, 'network.npz')
sparse.save_npz(network_path, nx.to_scipy_sparse_array(G))

# Plot degree distribution
plt.figure(figsize=(7,5))
plt.hist(deg_seq, bins=30, color='skyblue', edgecolor='k')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Degree Distribution (Boarding School SBM)')
deghist_path = os.path.join(output_dir, 'degree-distribution.png')
plt.tight_layout()
plt.savefig(deghist_path)
plt.close()

import pandas as pd
house_csv_path = os.path.join(output_dir, 'house-mapping.csv')
pd.DataFrame(list(node_house.items()), columns=['node','house']).to_csv(house_csv_path, index=False)

# Minimal code signature
code_path = os.path.join(output_dir, 'network-design.py')
with open(code_path, 'w') as f:
    f.write('# Constructed boarding school SBM with block sizes: ' + str(block_sizes) + '\n')
    f.write('# With p_in = {:.3f}, p_out = {:.3f}\n'.format(p_in, p_out))
    f.write('# Node-house assignments in CSV, adjacency in NPZ.\n')
    f.write('# Degree/centrality plots in PNG.\n')

result = {
    'diagnostics': {
        'mean_degree': mean_deg,
        'second_moment_degree': mean_sq_deg,
        'clustering': clustering,
        'assortativity': assort,
        'gcc_size': gcc_size,
        'n': n
    },
    'network_path': network_path,
    'plot_paths': {
        deghist_path: 'Degree distribution of contact network (SBM with block structure for houses)',
        house_csv_path: 'CSV of node-to-house assignment for stratified simulation analyses'
    },
    'code_path': code_path,
    'summary': f"Network: SBM with 11 blocks (houses). Block sizes: {block_sizes} | p_in={p_in}, p_out={p_out}. N={n}. Mean deg={mean_deg:.2f}, sec moment={mean_sq_deg:.2f}, clustering={clustering:.3f}, assort={assort:.3f}, GCC={gcc_size}."
}

result
