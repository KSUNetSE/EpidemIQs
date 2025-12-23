
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy import sparse

# Step 1: Define SBM parameters
sizes = [20, 80] # 20 core, 80 periphery
probs = [[0.5, 0.2], [0.2, 0.02]]
np.random.seed(42)  # for reproducibility

# Step 2: Generate SBM network
G = nx.stochastic_block_model(sizes, probs, seed=42)

# Step 3: Annotate nodes with 'core' or 'periphery' label
block_labels = ['core'] * sizes[0] + ['periphery'] * sizes[1]
for idx, label in enumerate(block_labels):
    G.nodes[idx]['block'] = label

# Step 4: Calculate blockwise average degrees
core_nodes = [n for n,d in G.nodes(data=True) if d['block']=='core']
peri_nodes = [n for n,d in G.nodes(data=True) if d['block']=='periphery']

deg_all = dict(G.degree())
avg_deg_core = np.mean([deg_all[n] for n in core_nodes])
avg_deg_peri = np.mean([deg_all[n] for n in peri_nodes])

# Step 5: Calculate global mean degree and second moment
all_degrees = np.array([deg for _, deg in G.degree()])
k_mean = np.mean(all_degrees)
k2_mean = np.mean(all_degrees**2)

# Step 6: Plot degree histogram colored by block
plt.figure(figsize=(6,4))
plt.hist([deg_all[n] for n in core_nodes], bins=15, alpha=0.7, label='Core', color='tab:blue')
plt.hist([deg_all[n] for n in peri_nodes], bins=15, alpha=0.7, label='Periphery', color='tab:orange')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.legend()
plt.title('Degree Distribution by Block')
degree_hist_path = os.path.join(os.getcwd(),"output", "degree-histogram-core-periphery.png")
plt.tight_layout()
plt.savefig(degree_hist_path)
plt.close()

# Step 7: Optional diagnostic: size of GCC
lcc = max(nx.connected_components(G), key=len)
GCC_size = len(lcc)

# Step 8: Save network
network_path = os.path.join(os.getcwd(), "output", "network-core-periphery-banking.npz")
sparse.save_npz(network_path, nx.to_scipy_sparse_array(G))

# Step 9: Assortativity (mixing by degree)
degree_assort = nx.degree_assortativity_coefficient(G)

# Step 10: Return details for observation:
results = {
    'avg_deg_core': avg_deg_core,
    'avg_deg_peri': avg_deg_peri,
    'k_mean': k_mean,
    'k2_mean': k2_mean,
    'GCC_size': GCC_size,
    'degree_assort': degree_assort,
    'network_path': network_path,
    'degree_hist_path': degree_hist_path
}
results