
import networkx as nx
import numpy as np
from scipy import sparse
import os

# ER network basics
N = 1000
k_mean = 10
p = k_mean / (N - 1)

# Construct ER network
erg = nx.erdos_renyi_graph(n=N, p=p, seed=42)

# Basic properties
degrees = [d for _, d in erg.degree()]
actual_k = np.mean(degrees)
k2 = np.mean(np.power(degrees,2))

# Giant component size
gcc = max(nx.connected_components(erg), key=len)
size_gcc = len(gcc)

# Connectivity
is_conn = nx.is_connected(erg)

# Save network
er_npz_path = os.path.join(os.getcwd(), "output", "erdosrenyi-n1000-k10.npz")
sparse.save_npz(er_npz_path, nx.to_scipy_sparse_array(erg))

return_vars = ['actual_k','k2','size_gcc','is_conn','er_npz_path','degrees']