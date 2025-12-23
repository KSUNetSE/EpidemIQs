
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import os
# Reuse sequence from previous code (degree sequence with k=2, k=3, k=10)
N = 10000
sequence = [2]*7500 + [3]*1429 + [10]*1071
np.random.shuffle(sequence)
# Guarantee even sum
if sum(sequence)%2 != 0:
    for i in range(len(sequence)):
        if sequence[i] > 0:
            sequence[i] -= 1
            break
# Build configuration model
G = nx.configuration_model(sequence, seed=42)
G = nx.Graph(G)
G.remove_edges_from(nx.selfloop_edges(G))
degrees = [d for n,d in G.degree()]
mean_deg = np.mean(degrees)
second_mom = np.mean(np.square(degrees))
q = (second_mom - mean_deg)/mean_deg
frac_10 = np.sum(np.array(degrees)==10)/N
# Diagnostics and plots
counts = np.bincount(degrees)
deg_vals = np.arange(len(counts))
plt.figure(figsize=(7,5))
plt.bar(deg_vals, counts, width=0.8, color='cornflowerblue')
plt.xlabel('Degree k')
plt.ylabel('Count')
plt.title('Degree Distribution (Empirical, Simple Config Model)')
plt.tight_layout()
deg_hist_path = os.path.join(os.getcwd(), "output", "deg-hist-configmodel.png")
plt.savefig(deg_hist_path)
plt.close()
# Save the network for future simulation
net_path = os.path.join(os.getcwd(), "output", "configmodel-z3-q4.npz")
sparse.save_npz(net_path, nx.to_scipy_sparse_array(G))
# Simple diagnostics: Assortativity, clustering, GCC size
assort = nx.degree_pearson_correlation_coefficient(G)
clust = nx.transitivity(G)
lcc = len(max(nx.connected_components(G), key=len))/N
return_vars = [
    'mean_deg', 'second_mom', 'q', 'frac_10',
    'deg_hist_path', 'net_path', 'assort', 'clust', 'lcc'
]