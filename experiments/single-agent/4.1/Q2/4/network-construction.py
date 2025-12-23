"""
Barabasi-Albert scale-free network construction (n=1000, m=4)
"""
import networkx as nx
from scipy import sparse
import numpy as np
n = 1000
m = 4
G = nx.barabasi_albert_graph(n, m, seed=42)
sparse.save_npz('network.npz', nx.to_scipy_sparse_array(G))
