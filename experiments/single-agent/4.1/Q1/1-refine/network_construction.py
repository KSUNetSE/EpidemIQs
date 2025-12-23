
# Plan:
# 1. Construct networks: one homogeneous (e.g. ER random graph with narrow degree distribution) and one degree-heterogeneous (Barabasi-Albert scale-free).
# 2. Save them for later simulation.
# 3. Calculate <k> and <k^2> for both.
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

N = 1000 # population size for clear stochastic/deterministic comparison
mean_degree = 10
# Homogeneous network: Erdős–Rényi (ER)
p_er = mean_degree / (N - 1)
G_er = nx.erdos_renyi_graph(n=N, p=p_er, seed=123)
# Degree-heterogeneous: Barabasi-Albert (BA)
m_ba = mean_degree // 2
G_ba = nx.barabasi_albert_graph(n=N, m=m_ba, seed=123)

# Save networks
sparse.save_npz(os.path.join(output_dir, 'network_er.npz'), nx.to_scipy_sparse_array(G_er))
sparse.save_npz(os.path.join(output_dir, 'network_ba.npz'), nx.to_scipy_sparse_array(G_ba))

# Degree distributions
deg_er = np.array([d for n, d in G_er.degree()])
deg_ba = np.array([d for n, d in G_ba.degree()])

plt.hist(deg_er, bins=range(0, max(deg_er)+2), alpha=0.7, label='ER')
plt.hist(deg_ba, bins=range(0, max(deg_ba)+2, 2), alpha=0.7, label='BA')
plt.legend()
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distributions: ER (homogeneous) vs BA (heterogeneous)')
plt.savefig(os.path.join(output_dir, 'degree_distribution.png'))
plt.close()

# Calculate moments
k_er = deg_er.mean()
k2_er = (deg_er**2).mean()
k_ba = deg_ba.mean()
k2_ba = (deg_ba**2).mean()

res = {
    'network_paths': [os.path.join(output_dir, 'network_er.npz'), os.path.join(output_dir, 'network_ba.npz')],
    'mean_k_er': k_er,
    'k2_er': k2_er,
    'mean_k_ba': k_ba,
    'k2_ba': k2_ba,
    'deg_dist_plot': os.path.join(output_dir, 'degree_distribution.png')
}
res