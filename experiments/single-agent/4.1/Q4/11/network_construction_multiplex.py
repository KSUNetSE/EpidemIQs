
import networkx as nx
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import os
os.makedirs(os.path.join(os.getcwd(), 'output'), exist_ok=True)

# Parameters for layers
n = 500
p_A = 0.015  # ER probability for layer A
p_B = 0.015  # ER probability for layer B (will permute to decorrelate)

# Layer A: ER graph
G_A = nx.erdos_renyi_graph(n, p_A, seed=42)
A_csr = nx.to_scipy_sparse_array(G_A)

def permute_graph_edges(G):
    # Permute node labels to reduce correlation with G_A
    mapping = dict(zip(G.nodes(), np.random.permutation(list(G.nodes()))))
    return nx.relabel_nodes(G, mapping)

# Layer B: independent ER, then permute
G_B = nx.erdos_renyi_graph(n, p_B, seed=43)
G_B = permute_graph_edges(G_B)
B_csr = nx.to_scipy_sparse_array(G_B)

# Save both networks
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network_A.npz'), A_csr)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network_B.npz'), B_csr)

# Compute degree stats for each
degrees_A = np.array([d for n, d in G_A.degree()])
kA = np.mean(degrees_A)
kA2 = np.mean(degrees_A**2)
degrees_B = np.array([d for n, d in G_B.degree()])
kB = np.mean(degrees_B)
kB2 = np.mean(degrees_B**2)

# Plot degree distributions
plt.figure(figsize=(8, 4))
plt.hist(degrees_A, bins=30, alpha=0.6, label='Layer A')
plt.hist(degrees_B, bins=30, alpha=0.6, label='Layer B')
plt.legend()
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Degree Distributions of Both Layers')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'degree_dist_AB.png'))

# Overlap of top-10 central nodes (by degree) in each layer
topA = set(np.argsort(-degrees_A)[:10])
topB = set(np.argsort(-degrees_B)[:10])
overlap_top = len(topA & topB)

result = {
    'network_paths': [os.path.join(os.getcwd(), 'output', 'network_A.npz'), os.path.join(os.getcwd(), 'output', 'network_B.npz')],
    'kA': kA,
    'kA2': kA2,
    'kB': kB,
    'kB2': kB2,
    'overlap_top_10': overlap_top,
    'plot_path': os.path.join(os.getcwd(), 'output', 'degree_dist_AB.png'),
    'n': n
}
result