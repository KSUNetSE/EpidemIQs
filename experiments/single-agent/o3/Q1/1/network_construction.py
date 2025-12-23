
import os, networkx as nx, numpy as np, scipy.sparse as sparse
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
N = 5000
# ER network
er_p = 8 / (N-1)
G_er = nx.fast_gnp_random_graph(N, er_p, seed=42, directed=False)
# BA network
m = 4
G_ba = nx.barabasi_albert_graph(N, m, seed=42)
# save networks
sparse.save_npz(os.path.join(output_dir, 'network_ER.npz'), nx.to_scipy_sparse_array(G_er))
sparse.save_npz(os.path.join(output_dir, 'network_BA.npz'), nx.to_scipy_sparse_array(G_ba))
# compute statistics
k_er = np.array([d for _, d in G_er.degree()])
mean_k_er = k_er.mean()
second_moment_er = (k_er**2).mean()

k_ba = np.array([d for _, d in G_ba.degree()])
mean_k_ba = k_ba.mean()
second_moment_ba = (k_ba**2).mean()

a = {'mean_k_er': mean_k_er, 'second_moment_er': second_moment_er, 'mean_k_ba': mean_k_ba, 'second_moment_ba': second_moment_ba}
