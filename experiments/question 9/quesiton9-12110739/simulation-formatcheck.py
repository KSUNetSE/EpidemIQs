
# There is a persistent coo/csr conversion issue: scipy.sparse.load_npz returns coo_array as of scipy >= 1.8,
# and coo_array.tocsr() returns *a new* csr matrix (not modifies in-place). Let's inspect this with a very minimal test right after loading.
import scipy.sparse
adj_path = '/Users/hosseinsamaei/phd/epidemiqs/output/er-adjacency.npz'
adj = scipy.sparse.load_npz(adj_path)
adj_csr = adj.tocsr()
check = (adj_csr.format, hasattr(adj_csr,'indices'), type(adj_csr).__name__, adj.shape)
return_vars=['check']