
import scipy.sparse as sparse, numpy as np, os, fastgemf as fg
A=sparse.load_npz(os.path.join(os.getcwd(),'output','layer_A.npz'))
# Generate correlated B by mixture of A and random
n=A.shape[0]
import networkx as nx
G_rand=nx.erdos_renyi_graph(n,0.01,seed=2)
R=sparse.csr_matrix(nx.to_scipy_sparse_array(G_rand))
alpha=0.3
B3=alpha*A+(1-alpha)*R
sparse.save_npz(os.path.join(os.getcwd(),'output','layer_B3.npz'),B3)
