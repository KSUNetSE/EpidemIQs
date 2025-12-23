
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

### PARAMETERS (can be easily tuned here)
N_total = 100
N_core = 20
N_periphery = 80
p_cc = 0.5      # intra-core probability
p_cp = 0.2      # core-periphery link prob
p_pp = 0.02     # periphery-periphery link prob
np.random.seed(42)

# Assign node sets
core_nodes = list(range(N_core))
periphery_nodes = list(range(N_core, N_total))

# Build network
G = nx.Graph()
G.add_nodes_from(range(N_total))
roles = {i: ("core" if i in core_nodes else "periphery") for i in range(N_total)}

# Add core-core edges
for i in core_nodes:
    for j in core_nodes:
        if i < j and np.random.rand() < p_cc:
            G.add_edge(i, j)

# Add periphery-periphery edges
for i in periphery_nodes:
    for j in periphery_nodes:
        if i < j and np.random.rand() < p_pp:
            G.add_edge(i, j)

# Add core-periphery edges
for i in core_nodes:
    for j in periphery_nodes:
        if np.random.rand() < p_cp:
            G.add_edge(i, j)

# Calculate diagnostics
kvals = [deg for n, deg in G.degree()]
mean_k = np.mean(kvals)
second_moment_k = np.mean(np.array(kvals)**2)

core_degrees = [G.degree(n) for n in core_nodes]
periphery_degrees = [G.degree(n) for n in periphery_nodes]
mean_core_deg = np.mean(core_degrees)
mean_periph_deg = np.mean(periphery_degrees)

try:
    gcc_size = len(max(nx.connected_components(G), key=len))
    is_connected = nx.is_connected(G)
except Exception as e:
    gcc_size = -1
    is_connected = False

clustering = nx.average_clustering(G)
degree_assortativity = nx.degree_assortativity_coefficient(G)

# Plots and visualization
outputdir = os.path.join(os.getcwd(), "output")
os.makedirs(outputdir, exist_ok=True)

# Degree distribution
plt.figure(figsize=(7,4))
plt.hist(kvals, bins=np.arange(0,max(kvals)+2)-0.5, color='C0', edgecolor='black')
plt.title('Degree Distribution of Core-Periphery Network')
plt.xlabel('Node degree')
plt.ylabel('Number of nodes')
deg_dist_path = os.path.join(outputdir, "fig-degree-distribution.png")
plt.tight_layout()
plt.savefig(deg_dist_path)
plt.close()

# Adjacency block-structure plot
adj = nx.to_numpy_array(G, nodelist=range(N_total))
plt.figure(figsize=(7,7))
plt.imshow(adj, interpolation='none', cmap='Greys')
plt.title('Adjacency Matrix (Block Structure)\nCore = top-left 20×20')
adj_matrix_path = os.path.join(outputdir, "fig-adjacency-matrix.png")
plt.tight_layout()
plt.savefig(adj_matrix_path)
plt.close()

# Save network
net_path = os.path.join(outputdir, "network-core-periphery.npz")
sparse.save_npz(net_path, nx.to_scipy_sparse_array(G))

return_vars = [
    'mean_k', 'second_moment_k', 'mean_core_deg', 'mean_periph_deg', 'gcc_size', 'is_connected',
    'clustering', 'degree_assortativity', 'net_path', 'deg_dist_path', 'adj_matrix_path'
]
