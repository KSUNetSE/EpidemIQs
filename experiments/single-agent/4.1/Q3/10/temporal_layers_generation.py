
import fastgemf as fg
import numpy as np
import os
from scipy import sparse

# For the temporal activity-driven network, we'll approximate by time-discrete switching of network layers
# For 1000 steps, at each time step, the active edges form the temporal layer for that step
# We'll now prepare a list of adjacency matrices, one per time step, from saved edge list

temporal_network_file = os.path.join(os.getcwd(), 'output', 'temporal_network_edges.npz')
edge_lists_per_t = np.load(temporal_network_file, allow_pickle=True)
edge_lists = [edge_lists_per_t[f'arr_{i}'] for i in range(len(edge_lists_per_t.files))]

# For each t, build an adjacency matrix in CSR format for FastGEMF (sparse, undirected)
from scipy.sparse import csr_matrix
layer_adjs = []
for t in range(1000):
    elist = edge_lists[t]
    edges = np.array(elist)
    if len(edges) == 0:
        A = csr_matrix((1000, 1000), dtype=int)
    else:
        A = csr_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])), shape=(1000, 1000))
        A = A + A.T  # undirected adjacency
        A.data[:] = 1
    layer_adjs.append(A)

# Save all layers in one npz
for t, A in enumerate(layer_adjs):
    sparse.save_npz(os.path.join(os.getcwd(), f"output/temporal_layer_{t}.npz"), A)
'temporal_layers_created'