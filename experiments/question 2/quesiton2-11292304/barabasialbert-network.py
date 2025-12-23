
import networkx as nx
import numpy as np
from scipy import sparse
import os

# BA network basics
N = 1000
m = 5  # For average degree 2m = 10

# Construct BA network
bag = nx.barabasi_albert_graph(n=N, m=m, seed=42)

# Basic properties
degrees_ba = [d for _, d in bag.degree()]
actual_k_ba = np.mean(degrees_ba)
k2_ba = np.mean(np.power(degrees_ba,2))

# GCC size
gcc_ba = max(nx.connected_components(bag), key=len)
size_gcc_ba = len(gcc_ba)

# Connectivity
is_conn_ba = nx.is_connected(bag)

# Save network
ba_npz_path = os.path.join(os.getcwd(), "output", "barabasialbert-n1000-k10.npz")
sparse.save_npz(ba_npz_path, nx.to_scipy_sparse_array(bag))

return_vars = ['actual_k_ba','k2_ba','size_gcc_ba','is_conn_ba','ba_npz_path','degrees_ba']