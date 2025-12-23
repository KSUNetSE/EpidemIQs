
import networkx as nx
import numpy as np
from scipy import sparse
import os
# Load BA network
from scipy.sparse import load_npz
adj = load_npz(os.path.join(os.getcwd(), "output", "barabasialbert-n1000-k10.npz"))
bag = nx.from_scipy_sparse_array(adj)

# Centralities
ba_clustering = nx.average_clustering(bag)
ba_assortativity = nx.degree_assortativity_coefficient(bag)

# Degree moments
k_arr_ba = np.array([d for n,d in bag.degree()])
actual_k_ba = np.mean(k_arr_ba)
k2_ba = np.mean(k_arr_ba ** 2)
size_gcc_ba = len(max(nx.connected_components(bag), key=len))
is_conn_ba = nx.is_connected(bag)

return_vars = ['actual_k_ba','k2_ba', 'ba_clustering', 'ba_assortativity', 'size_gcc_ba', 'is_conn_ba']