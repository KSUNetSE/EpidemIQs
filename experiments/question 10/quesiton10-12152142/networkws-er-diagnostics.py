
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from scipy import sparse

# Ensure output directory exists
outdir = os.path.join(os.getcwd(), 'output')
os.makedirs(outdir, exist_ok=True)

# Parameters
N = 1000  # number of nodes
k = 6     # mean degree
# -- Watts-Strogatz parameters
p_sw = 0.07  # rewiring probability (small-world regime, high clustering, some shortcuts)
# -- Erdos-Renyi parameters
p_er = k / (N-1)

# Build networks
G_ws = nx.watts_strogatz_graph(N, k, p_sw)
G_er = nx.erdos_renyi_graph(N, p_er)

# Define functions for statistics
def mean_degree(G):
    return np.mean([d for n, d in G.degree()])
def sec_moment_degree(G):
    return np.mean([d ** 2 for n, d in G.degree()])

# Centralities/statistics
md_ws = mean_degree(G_ws)
md2_ws = sec_moment_degree(G_ws)
cluster_ws = nx.average_clustering(G_ws)
GCC_ws = len(max(nx.connected_components(G_ws), key=len))
assort_ws = nx.degree_pearson_correlation_coefficient(G_ws)

md_er = mean_degree(G_er)
md2_er = sec_moment_degree(G_er)
cluster_er = nx.average_clustering(G_er)
GCC_er = len(max(nx.connected_components(G_er), key=len))
assort_er = nx.degree_pearson_correlation_coefficient(G_er)

# Plots
fig1, ax1 = plt.subplots()
degseq_ws = [d for n, d in G_ws.degree()]
ax1.hist(degseq_ws, bins=range(min(degseq_ws), max(degseq_ws)+2), color='b', alpha=0.7, rwidth=0.85)
ax1.set_title('Degree Distribution: Watts-Strogatz')
ax1.set_xlabel('Degree')
ax1.set_ylabel('Frequency')
ws_plot_path = os.path.join(outdir, 'degree-distribution-ws.png')
fig1.savefig(ws_plot_path)
plt.close(fig1)

fig2, ax2 = plt.subplots()
degseq_er = [d for n, d in G_er.degree()]
ax2.hist(degseq_er, bins=range(min(degseq_er), max(degseq_er)+2), color='g', alpha=0.7, rwidth=0.85)
ax2.set_title('Degree Distribution: Erdos-Renyi')
ax2.set_xlabel('Degree')
ax2.set_ylabel('Frequency')
er_plot_path = os.path.join(outdir, 'degree-distribution-er.png')
fig2.savefig(er_plot_path)
plt.close(fig2)

# Save networks
ws_save_path = os.path.join(outdir, "network-watts-strogatz.npz")
er_save_path = os.path.join(outdir, "network-erdos-renyi.npz")
sparse.save_npz(ws_save_path, nx.to_scipy_sparse_array(G_ws))
sparse.save_npz(er_save_path, nx.to_scipy_sparse_array(G_er))

# Return stats and filepaths
ws_stats = {
    'mean_degree': md_ws, 'sec_moment_degree': md2_ws,
    'clustering': cluster_ws, 'giant_component_size': GCC_ws,
    'assortativity': assort_ws,
    'plot_path': ws_plot_path
}
er_stats = {
    'mean_degree': md_er, 'sec_moment_degree': md2_er,
    'clustering': cluster_er, 'giant_component_size': GCC_er,
    'assortativity': assort_er,
    'plot_path': er_plot_path
}
return_vars = ['ws_save_path', 'er_save_path', 'ws_stats', 'er_stats']
