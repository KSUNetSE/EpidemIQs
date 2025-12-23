
import os, numpy as np, scipy.sparse as sparse
from scipy.sparse.linalg import eigs

output_dir = os.path.join(os.getcwd(), 'output')
A_csr = sparse.load_npz(os.path.join(output_dir, 'network_A.npz'))
B_csr = sparse.load_npz(os.path.join(output_dir, 'network_B.npz'))

def leading_eigenvector(csr):
    eigval, eigvec = eigs(csr.astype(float), k=1, which='LR')
    vec = np.abs(eigvec[:,0].real)
    vec /= np.linalg.norm(vec)
    return vec

vA = leading_eigenvector(A_csr)
vB = leading_eigenvector(B_csr)

# Create permutation that pairs high vA with low original vB values
idx_vA_desc = np.argsort(-vA)  # high to low
idx_vB_asc = np.argsort(vB)    # low to high
perm = np.empty_like(idx_vA_desc)
perm[idx_vA_desc] = idx_vB_asc  # node i (in sorted vA order) maps to corresponding low vB node

# Apply permutation to B
B_low_corr = B_csr[perm,:][:,perm]

vB_low = leading_eigenvector(B_low_corr)
corr_low = float(np.dot(vA, vB_low))

sparse.save_npz(os.path.join(output_dir, 'network_B_lowcorr.npz'), B_low_corr)

corr_low