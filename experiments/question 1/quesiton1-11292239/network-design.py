
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
from scipy import sparse

# Parameters
N = 2000  # network size in both cases
mean_k = 8  # target mean degree
p_er = mean_k / (N-1)  # ER probability
m_ba = mean_k // 2  # for BA, average degree ~= 2*m

output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

# ER network generation
G_er = nx.gnp_random_graph(N, p_er)
er_adj = nx.to_scipy_sparse_array(G_er)
sparse.save_npz(os.path.join(output_dir, "er-network.npz"), er_adj)

# BA network generation
G_ba = nx.barabasi_albert_graph(N, m_ba)
ba_adj = nx.to_scipy_sparse_array(G_ba)
sparse.save_npz(os.path.join(output_dir, "ba-network.npz"), ba_adj)

# Diagnostics and plots
# --- ER
k_er = np.array([d for n, d in G_er.degree()])
k_mean_er = np.mean(k_er)
k2_mean_er = np.mean(k_er**2)
GCC_er = max(nx.connected_components(G_er), key=len)
GCC_size_er = len(GCC_er)

plt.figure()
plt.hist(k_er, bins=range(0, np.max(k_er)+2), density=True, alpha=0.7, color='steelblue')
plt.xlabel('Degree (k)')
plt.ylabel('P(k)')
plt.title('ER Degree Distribution (N=2000, <k>=%.2f)' % k_mean_er)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot-er-degrees.png"))
plt.close()

# --- BA
k_ba = np.array([d for n, d in G_ba.degree()])
k_mean_ba = np.mean(k_ba)
k2_mean_ba = np.mean(k_ba**2)
GCC_ba = max(nx.connected_components(G_ba), key=len)
GCC_size_ba = len(GCC_ba)

plt.figure()
plt.hist(k_ba, bins=np.arange(np.min(k_ba), np.max(k_ba)+2)-0.5, density=True, alpha=0.7, color='orangered')
plt.xlabel('Degree (k)')
plt.ylabel('P(k)')
plt.title('BA Degree Distribution (N=2000, <k>=%.2f)' % k_mean_ba)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot-ba-degrees.png"))
plt.close()

# Final diagnostics compact return
nx_er_info = {
    "mean_degree": k_mean_er,
    "second_degree_moment": k2_mean_er,
    "gcc_size": GCC_size_er
}
nx_ba_info = {
    "mean_degree": k_mean_ba,
    "second_degree_moment": k2_mean_ba,
    "gcc_size": GCC_size_ba
}

return_vars = {
 'nx_er_info': nx_er_info, 'nx_ba_info': nx_ba_info,
}
