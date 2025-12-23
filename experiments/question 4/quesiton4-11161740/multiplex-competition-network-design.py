
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import pearsonr
from numpy.linalg import eig

def leading_eigenvector(adj):
    w, v = np.linalg.eig(adj)
    idx = np.argmax(np.abs(w))
    return np.abs(v[:, idx]) / np.linalg.norm(v[:, idx])

def degree_vector(G):
    return np.array([d for n, d in G.degree()])

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# --- Parameters ---
N = 1000
# High-correlation BA layers (case 1)
m_A = 4
m_B = 4
# Low-correlation: keep BA, but minimize hub overlap
# Construction

# Case 1: High overlap/correlation
np.random.seed(42)
G_A_hi = nx.barabasi_albert_graph(N, m_A, seed=42)
G_B_hi = G_A_hi.copy()
# To avoid being identical, rewire a small fraction (e.g. 5%)
num_edges = G_A_hi.number_of_edges()
rewire_fraction = 0.05
num_rewire = int(num_edges * rewire_fraction)
all_edges = list(G_B_hi.edges())
rewired = 0
for (u, v) in np.random.permutation(all_edges):
    if rewired >= num_rewire:
        break
    G_B_hi.remove_edge(u, v)
    candidates = set(range(N)) - set([u]) - set(G_B_hi[u])
    if len(candidates) < 1:
        continue
    new_v = np.random.choice(list(candidates))
    G_B_hi.add_edge(u, new_v)
    rewired += 1

# Case 2: Low overlap/correlation
G_A_lo = nx.barabasi_albert_graph(N, m_A, seed=43)
# For B: generate a new BA, then permute node labels
G_B_dummy = nx.barabasi_albert_graph(N, m_B, seed=44)
perm = np.random.permutation(N)
G_B_lo = nx.relabel_nodes(G_B_dummy, {i: int(perm[i]) for i in range(N)})

# Check degree and eigenvector correlations
edges_A_hi = nx.to_scipy_sparse_array(G_A_hi)
edges_B_hi = nx.to_scipy_sparse_array(G_B_hi)
edges_A_lo = nx.to_scipy_sparse_array(G_A_lo)
edges_B_lo = nx.to_scipy_sparse_array(G_B_lo)

def network_metrics(G, adj):
    degvec = degree_vector(G)
    mean_deg = degvec.mean()
    mean_deg2 = (degvec ** 2).mean()
    # GCC size
    if nx.is_directed(G):
        gcc = max((len(c) for c in nx.strongly_connected_components(G)), default=0)
    else:
        gcc = max((len(c) for c in nx.connected_components(G)), default=0)
    # Largest eigenvalue
    lamb1 = np.max(np.abs(np.linalg.eigvals(adj.toarray())))
    return mean_deg, mean_deg2, gcc, lamb1, degvec

res_A_hi = network_metrics(G_A_hi, edges_A_hi)
res_B_hi = network_metrics(G_B_hi, edges_B_hi)
res_A_lo = network_metrics(G_A_lo, edges_A_lo)
res_B_lo = network_metrics(G_B_lo, edges_B_lo)

# Degree & eigenvector correlation (important diagnostics)
deg_corr_hi, _ = pearsonr(res_A_hi[4], res_B_hi[4])
eigvec_A_hi = leading_eigenvector(edges_A_hi.toarray())
eigvec_B_hi = leading_eigenvector(edges_B_hi.toarray())
eig_corr_hi = cosine_similarity(eigvec_A_hi, eigvec_B_hi)

deg_corr_lo, _ = pearsonr(res_A_lo[4], res_B_lo[4])
eigvec_A_lo = leading_eigenvector(edges_A_lo.toarray())
eigvec_B_lo = leading_eigenvector(edges_B_lo.toarray())
eig_corr_lo = cosine_similarity(eigvec_A_lo, eigvec_B_lo)

# --- Save adjacency matrices ---
outputdir = os.path.join(os.getcwd(), "output")
os.makedirs(outputdir, exist_ok=True)
sparse.save_npz(os.path.join(outputdir, "layerA-highoverlap.npz"), edges_A_hi)
sparse.save_npz(os.path.join(outputdir, "layerB-highoverlap.npz"), edges_B_hi)
sparse.save_npz(os.path.join(outputdir, "layerA-lowoverlap.npz"), edges_A_lo)
sparse.save_npz(os.path.join(outputdir, "layerB-lowoverlap.npz"), edges_B_lo)

# --- Diagnostic Plots ---
plt.figure(figsize=(8,4))
plt.hist(res_A_hi[4], bins=40, alpha=0.7, label="Layer A Hi-overlap")
plt.hist(res_B_hi[4], bins=40, alpha=0.7, label="Layer B Hi-overlap")
plt.legend(); plt.xlabel("Degree"); plt.ylabel("Num nodes"); plt.tight_layout()
plt.savefig(os.path.join(outputdir, "degdist-highoverlap.png"))
plt.close()

plt.figure(figsize=(8,4))
plt.hist(res_A_lo[4], bins=40, alpha=0.7, label="Layer A Low-overlap")
plt.hist(res_B_lo[4], bins=40, alpha=0.7, label="Layer B Low-overlap")
plt.legend(); plt.xlabel("Degree"); plt.ylabel("Num nodes"); plt.tight_layout()
plt.savefig(os.path.join(outputdir, "degdist-lowoverlap.png"))
plt.close()

# --- Save centrality vectors for analysis (optional) ---
np.savez(os.path.join(outputdir, "eigvecs-highoverlap.npz"), eig_A=eigvec_A_hi, eig_B=eigvec_B_hi)
np.savez(os.path.join(outputdir, "eigvecs-lowoverlap.npz"), eig_A=eigvec_A_lo, eig_B=eigvec_B_lo)

results = {
    'hi': {
        'layerA_metrics': res_A_hi,
        'layerB_metrics': res_B_hi,
        'deg_corr': deg_corr_hi,
        'eig_corr': eig_corr_hi
    },
    'lo': {
        'layerA_metrics': res_A_lo,
        'layerB_metrics': res_B_lo,
        'deg_corr': deg_corr_lo,
        'eig_corr': eig_corr_lo
    },
    'paths': {
        'layerA_high': os.path.join(outputdir, "layerA-highoverlap.npz"),
        'layerB_high': os.path.join(outputdir, "layerB-highoverlap.npz"),
        'layerA_low': os.path.join(outputdir, "layerA-lowoverlap.npz"),
        'layerB_low': os.path.join(outputdir, "layerB-lowoverlap.npz"),
        'degdist_high': os.path.join(outputdir, "degdist-highoverlap.png"),
        'degdist_low': os.path.join(outputdir, "degdist-lowoverlap.png")
    }
}
