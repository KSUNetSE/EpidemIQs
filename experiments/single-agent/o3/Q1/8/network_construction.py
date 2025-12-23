
import os, networkx as nx, scipy.sparse as sparse, numpy as np
from collections import defaultdict

# Ensure output directory exists
out_dir = os.path.join(os.getcwd(), 'output')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

N = 5000
mean_deg_target = 10

# 1. Erdős-Rényi (homogeneous-like) network
p_er = mean_deg_target / (N - 1)
G_er = nx.fast_gnp_random_graph(N, p_er, seed=42)

# 2. Barabási–Albert (degree-heterogeneous) network
m_ba = mean_deg_target // 2  # Each new node attaches to m=5 existing nodes -> avg deg ~ 2m
G_ba = nx.barabasi_albert_graph(N, m_ba, seed=42)

# Save the networks
sparse.save_npz(os.path.join(out_dir, 'network_ER.npz'), nx.to_scipy_sparse_array(G_er, dtype=np.int8))
sparse.save_npz(os.path.join(out_dir, 'network_BA.npz'), nx.to_scipy_sparse_array(G_ba, dtype=np.int8))

# Compute statistics
stats = {}
for name, G in [('ER', G_er), ('BA', G_ba)]:
    degrees = np.array([d for _, d in G.degree()])
    k_mean = degrees.mean()
    k2_mean = (degrees**2).mean()
    stats[name] = {'k_mean': k_mean, 'k2_mean': k2_mean}
print(stats)
