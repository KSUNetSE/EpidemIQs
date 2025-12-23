
# The issues stem from scipy.sparse.csr_array, which is *not* fully compatible with csr_matrix API (lack of .getrow and sometimes odd behavior).
# We'll directly convert to csr_matrix for this simulation. This fixes all API differences and is safest for mature code.

import scipy.sparse
adj_path = '/Users/hosseinsamaei/phd/epidemiqs/output/er-adjacency.npz'
adj = scipy.sparse.load_npz(adj_path)
from scipy.sparse import csr_matrix
if not isinstance(adj, csr_matrix):
    adj = csr_matrix(adj)
# Now we can use adj.getrow(0).indices
indices0 = adj.getrow(0).indices
indices1 = adj.getrow(1).indices
return_vars = ['indices0','indices1']