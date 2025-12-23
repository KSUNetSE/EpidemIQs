
# Check types and ensure all elements in layer_seq for simulation are correct CSR matrices, for debugging
from scipy.sparse import load_npz, issparse
import os
layer_seq_debug = [load_npz(os.path.join(os.getcwd(), f"output/temporal_layer_{t}.npz")) for t in range(10)]
all_csr = all(issparse(A) for A in layer_seq_debug)
str([A.shape for A in layer_seq_debug]), all_csr, str(type(layer_seq_debug[0]))