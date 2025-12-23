
import os, numpy as np, networkx as nx, scipy.sparse as sparse
from scipy.sparse.linalg import eigs
import random

output_dir = os.path.join(os.getcwd(), 'output')

A_csr = sparse.load_npz(os.path.join(output_dir, 'network_A.npz'))
B_csr = sparse.load_npz(os.path.join(output_dir, 'network_B.npz'))

N = A_csr.shape[0]

# Compute leading eigenvectors (normalized)
def leading_eigenvector(csr):
    eigval, eigvec = eigs(csr.astype(float), k=1, which='LR')
    v = np.abs(eigvec[:,0].real)
    v = v / np.linalg.norm(v)
    return v

vA = leading_eigenvector(A_csr)

# identity B (ER) eigenvector
vB = leading_eigenvector(B_csr)

corr_original = np.dot(vA, vB)

# create permuted version of B: random shuffle nodes
perm = np.random.RandomState(42).permutation(N)
B_perm_csr = B_csr[perm,:][:,perm]

vB_perm = leading_eigenvector(B_perm_csr)

corr_perm = np.dot(vA, vB_perm)

# save B_perm network
sparse.save_npz(os.path.join(output_dir, 'network_B_perm.npz'), B_perm_csr)

# identical layers case: B_same = A_csr (just copy path)
sparse.save_npz(os.path.join(output_dir, 'network_B_same.npz'), A_csr)

# return correlations
corrs = {'corr_original': float(corr_original), 'corr_perm': float(corr_perm)}
