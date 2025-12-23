
# network-summary-static.py v2, using correct API for scipy.sparse csr_array
import numpy as np
import scipy.sparse as sp
import os

static_net_path = os.path.join(os.getcwd(), 'output', 'exp3-static-weighted.npz')
G_csr = sp.load_npz(static_net_path)
N = G_csr.shape[0]

# Use correct property: csr_matrix has .getnnz (rows), csr_array .nnz is total nonzero
degrees = np.diff(G_csr.indptr)  # number of nonzero per row
strengths = np.array(G_csr.sum(axis=1)).flatten()
edge_weights = G_csr.data
summary = {
    'N': N,
    'num_edges': int(G_csr.nnz//2),  # undirected
    'mean_degree': float(np.mean(degrees)),
    'var_degree': float(np.var(degrees)),
    'mean_strength': float(np.mean(strengths)),
    'var_strength': float(np.var(strengths)),
    'min_edge_weight': float(edge_weights.min()),
    'max_edge_weight': float(edge_weights.max()),
    'mean_edge_weight': float(edge_weights.mean()),
    'edge_weight_std': float(edge_weights.std()),
    'min_strength': float(strengths.min()),
    'max_strength': float(strengths.max()),
    'summary_complete': True
}
