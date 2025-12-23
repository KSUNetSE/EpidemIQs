
# Print all computed ER stats again for record, as a return statement
import networkx as nx
import numpy as np
from scipy.sparse import load_npz
import os
net = nx.from_scipy_sparse_array(load_npz(os.path.join(os.getcwd(), "output", "erdosrenyi-n1000-k10.npz")))
deg = np.array([d for n,d in net.degree()])
er_centrality = nx.average_clustering(net)
er_assort = nx.degree_assortativity_coefficient(net)
k_mean = deg.mean()
k2 = (deg**2).mean()
gcc = max(nx.connected_components(net), key=len)
gcc_size = len(gcc)
is_conn = nx.is_connected(net)
return_vars = ["k_mean","k2","er_centrality","er_assort","gcc_size","is_conn"]