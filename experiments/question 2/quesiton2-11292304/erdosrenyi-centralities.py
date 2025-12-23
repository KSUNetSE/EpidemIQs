
import networkx as nx
import numpy as np
from scipy import sparse
import os

# Load ER network
from scipy.sparse import load_npz
er_adj = load_npz(os.path.join(os.getcwd(), "output", "erdosrenyi-n1000-k10.npz"))
erg = nx.from_scipy_sparse_array(er_adj)

# Centralities and properties
er_clustering = nx.average_clustering(erg)
er_assortativity = nx.degree_assortativity_coefficient(erg)

# Degree moments
k_arr = np.array([d for n,d in erg.degree()])
actual_k = np.mean(k_arr)
k2 = np.mean(k_arr ** 2)
size_gcc = len(max(nx.connected_components(erg), key=len))
is_conn = nx.is_connected(erg)

return_vars = ['actual_k','k2', 'er_clustering', 'er_assortativity', 'size_gcc', 'is_conn']