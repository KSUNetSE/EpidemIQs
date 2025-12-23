
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy import sparse

# Use the analytic solution: 1125 nodes at degree 10, 8875 at degree 2
N = 10000
n10 = 1125
n2 = 8875

degree_sequence = [10]*n10 + [2]*n2
if sum(degree_sequence) % 2 != 0:
    degree_sequence[0] = 3  # adjust first node to degree 3 in order to make sum even
    # Now 1 node at degree 3, 1124 at 10, 8875-1=8874 at 2

degree_sequence = np.array(degree_sequence)
np.random.shuffle(degree_sequence)

G = nx.configuration_model(degree_sequence, seed=42)
G = nx.Graph(G)
G.remove_edges_from(nx.selfloop_edges(G))

k_list = [d for n, d in G.degree()]
mean_k = np.mean(k_list)
k_sq = np.mean(np.array(k_list)**2)
mean_excess_k = (k_sq - mean_k) / mean_k

deg_hist_path = os.path.join(os.getcwd(), 'output', 'degree-distribution.png')
plt.hist(k_list, bins=range(1, max(k_list) + 2), align='left', color='tab:blue', edgecolor='k')
plt.xlabel('Degree k')
plt.ylabel('Number of nodes')
plt.title('Degree Distribution')
plt.tight_layout()
plt.savefig(deg_hist_path)
plt.close()

# Save network
net_path = os.path.join(os.getcwd(), 'output', 'network.npz')
sparse.save_npz(net_path, nx.to_scipy_sparse_array(G))

GCC = max(nx.connected_components(G), key=len)
GCC_fraction = len(GCC) / N
try:
    assortativity = nx.degree_pearson_correlation_coefficient(G)
except Exception:
    assortativity = float('nan')
summary = {'mean_k': mean_k, 'mean_excess_k': mean_excess_k, 'n10': n10,
           'n2': n2, 'GCC_fraction': GCC_fraction, 'assortativity': assortativity,
           'net_path': net_path, 'deg_hist_path': deg_hist_path}
summary
