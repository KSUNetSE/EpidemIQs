
import os, networkx as nx, numpy as np, scipy.sparse as sparse
from scipy.sparse.linalg import eigs

output_dir = os.path.join(os.getcwd(), 'output')
N = 1000
d = 6
G_reg = nx.random_regular_graph(d, N, seed=2024)
B_reg = nx.to_scipy_sparse_array(G_reg, format='csr', dtype=float)
sparse.save_npz(os.path.join(output_dir, 'network_B_reg.npz'), B_reg)

def leading_vec(csr):
    eigval, eigvec = eigs(csr.astype(float), k=1, which='LR')
    v = np.abs(eigvec[:,0].real)
    v/=np.linalg.norm(v)
    return v

A_csr = sparse.load_npz(os.path.join(output_dir, 'network_A.npz'))
vA = leading_vec(A_csr)
vB_reg = leading_vec(B_reg)
corr_reg = float(np.dot(vA, vB_reg))
