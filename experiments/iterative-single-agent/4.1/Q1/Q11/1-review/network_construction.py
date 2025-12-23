
import networkx as nx
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
import os

# Construct two networks: 
# 1) Homogeneous-mixing (Erdős-Rényi random graph with narrow degree)
# 2) Degree-heterogeneous (Barabási-Albert scale-free)

N = 1000
# 1. ER graph with mean degree k = 8
kmean_er = 8
p_er = kmean_er / (N-1)
G_er = nx.erdos_renyi_graph(N, p_er, seed=42)

# 2. Barabási-Albert scale-free with mean degree ~ 8
G_ba = nx.barabasi_albert_graph(N, int(kmean_er/2), seed=42)

# Save networks
outdir = os.path.join(os.getcwd(), 'output')
if not os.path.exists(outdir):
    os.makedirs(outdir)
sparse.save_npz(os.path.join(outdir, "network-er.npz"), nx.to_scipy_sparse_array(G_er))
sparse.save_npz(os.path.join(outdir, "network-ba.npz"), nx.to_scipy_sparse_array(G_ba))

# Calculate <k> and <k^2>
def degree_moments(G):
    degrees = np.array([d for n, d in G.degree()])
    kmean = np.mean(degrees)
    k2 = np.mean(degrees**2)
    return kmean, k2
kmean_er, k2_er = degree_moments(G_er)
kmean_ba, k2_ba = degree_moments(G_ba)

# Plot degree distributions
plt.figure(figsize=(8,4))
plt.hist([d for n,d in G_er.degree()], bins=range(0, max(dict(G_er.degree()).values())+1), alpha=0.6, label="ER")
plt.hist([d for n,d in G_ba.degree()], bins=range(0, max(dict(G_ba.degree()).values())+1), alpha=0.6, label="BA")
plt.xlabel('Degree')
plt.ylabel('Count')
plt.legend()
plt.title('Degree Distributions: ER vs BA')
plt.tight_layout()
plt.savefig(os.path.join(outdir, "degree-dist-er-vs-ba.png"))

paths = {
    "er": os.path.join(outdir, "network-er.npz"),
    "ba": os.path.join(outdir, "network-ba.npz"),
    "plot": os.path.join(outdir, "degree-dist-er-vs-ba.png")
}
(kmean_er, k2_er), (kmean_ba, k2_ba), paths