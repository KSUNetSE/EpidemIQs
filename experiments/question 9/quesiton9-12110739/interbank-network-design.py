
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import sparse

# ---- Parameters ----
N = 500                    # Population size (number of banks)
z = 20                     # Target mean degree for ER (diversified, high degree)
m = 3                      # Number of links per new node for BA (concentrated, low degree)
E = 100.0                  # Total interbank exposure per bank (arbitrary units, same for all)
B = 20.0                   # Capital buffer per bank
phi = B/E                  # Default threshold (not directly used, just computing for context)
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

# ---- Diversified (ER) Network Construction ----
p = z / (N - 1)
G_ER = nx.erdos_renyi_graph(N, p, seed=42)

# Assign/Store edge weights: for each node, edges get weight = E/degree
for node in G_ER.nodes():
    k = G_ER.degree[node]
    if k > 0:
        for neighbor in G_ER[node]:
            G_ER[node][neighbor]['exposure'] = E / k

# ---- Concentrated (BA) Network Construction ----
G_BA = nx.barabasi_albert_graph(N, m, seed=42)

for node in G_BA.nodes():
    k = G_BA.degree[node]
    if k > 0:
        for neighbor in G_BA[node]:
            G_BA[node][neighbor]['exposure'] = E / k

# ---- Save Networks as Sparse Matrices ----
# Save adjacency (sparse) matrices
A_ER = nx.to_scipy_sparse_array(G_ER, weight=None, dtype=np.float32)
adj_ER_path = os.path.join(output_dir, "er-adjacency.npz")
sparse.save_npz(adj_ER_path, A_ER)

A_BA = nx.to_scipy_sparse_array(G_BA, weight=None, dtype=np.float32)
adj_BA_path = os.path.join(output_dir, "ba-adjacency.npz")
sparse.save_npz(adj_BA_path, A_BA)

# Save edge weight matrices as sparse arrays (exposures)
W_ER = nx.to_scipy_sparse_array(G_ER, weight='exposure', dtype=np.float32)
weight_ER_path = os.path.join(output_dir, "er-exposures.npz")
sparse.save_npz(weight_ER_path, W_ER)

W_BA = nx.to_scipy_sparse_array(G_BA, weight='exposure', dtype=np.float32)
weight_BA_path = os.path.join(output_dir, "ba-exposures.npz")
sparse.save_npz(weight_BA_path, W_BA)

# ---- Centralities and Metrics ----
def degree_moments(G):
    deg_seq = [G.degree[n] for n in G.nodes]
    mean_k = np.mean(deg_seq)
    mean_k2 = np.mean(np.square(deg_seq))
    return mean_k, mean_k2, deg_seq

mean_k_ER, mean_k2_ER, deg_seq_ER = degree_moments(G_ER)
mean_k_BA, mean_k2_BA, deg_seq_BA = degree_moments(G_BA)

# ---- Diagnostics: Degree Distributions ----
plt.figure(figsize=(7,4))
plt.hist(deg_seq_ER, bins=30, alpha=0.7, label='ER (diversified)', color='steelblue')
plt.xlabel('Degree (k)')
plt.ylabel('Number of Banks')
plt.title('Degree Distribution (ER)')
plt.legend()
deg_hist_ER_path = os.path.join(output_dir, 'er-degree-histogram.png')
plt.tight_layout()
plt.savefig(deg_hist_ER_path)
plt.close()

plt.figure(figsize=(7,4))
plt.hist(deg_seq_BA, bins=30, alpha=0.7, label='BA (concentrated)', color='orangered')
plt.xlabel('Degree (k)')
plt.ylabel('Number of Banks')
plt.title('Degree Distribution (BA)')
plt.legend()
deg_hist_BA_path = os.path.join(output_dir, 'ba-degree-histogram.png')
plt.tight_layout()
plt.savefig(deg_hist_BA_path)
plt.close()

# ---- Diagnostics: Giant Connected Component ----
lcc_ER = len(max(nx.connected_components(G_ER), key=len)) / N
lcc_BA = len(max(nx.connected_components(G_BA), key=len)) / N

# ---- Diagnostics: Clustering (optional) ----
gcc_ER = nx.transitivity(G_ER)
gcc_BA = nx.transitivity(G_BA)

# Output for network scientist
out_vars = {
    # network paths
    "adj_ER_path": adj_ER_path,
    "adj_BA_path": adj_BA_path,
    "weight_ER_path": weight_ER_path,
    "weight_BA_path": weight_BA_path,
    # diagnostic plot paths
    "deg_hist_ER_path": deg_hist_ER_path,
    "deg_hist_BA_path": deg_hist_BA_path,
    # key metrics
    "mean_k_ER": mean_k_ER,
    "mean_k2_ER": mean_k2_ER,
    "mean_k_BA": mean_k_BA,
    "mean_k2_BA": mean_k2_BA,
    # other diagnostics
    "lcc_ER": lcc_ER,
    "lcc_BA": lcc_BA,
    "gcc_ER": gcc_ER,
    "gcc_BA": gcc_BA
}

out_vars