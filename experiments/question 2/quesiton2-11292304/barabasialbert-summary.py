
# Print all computed BA stats again for record, as a return statement
import networkx as nx
import numpy as np
from scipy.sparse import load_npz
import os
ba = nx.from_scipy_sparse_array(load_npz(os.path.join(os.getcwd(), "output", "barabasialbert-n1000-k10.npz")))
deg_ba = np.array([d for n,d in ba.degree()])
ba_centrality = nx.average_clustering(ba)
ba_assort = nx.degree_assortativity_coefficient(ba)
k_mean_ba = deg_ba.mean()
k2_ba = (deg_ba**2).mean()
gcc_ba = max(nx.connected_components(ba), key=len)
size_gcc_ba = len(gcc_ba)
is_conn_ba = nx.is_connected(ba)
return_vars = ["k_mean_ba","k2_ba","ba_centrality","ba_assort","size_gcc_ba","is_conn_ba"]