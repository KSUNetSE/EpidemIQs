
import fastgemf as fg
import scipy.sparse as sparse
import os
import pandas as pd

# --- Step 2: Load network (ensure correct path and integrity for FastGEMF)
network_output_path = os.path.join(os.getcwd(), 'output', 'watts-strogatz-graph.npz')

# In case the output/output directory doesn't have network npz (if running outside specified user path),
# fallback: copy or, if really unavailable, create WS network (for reproducibility, alert if done)
if not os.path.exists(network_output_path):
    try:
        # Try to copy from provided absolute path
        import shutil
        shutil.copy('/Users/hosseinsamaei/phd/epidemiqs/output/watts-strogatz-graph.npz', network_output_path)
    except Exception as e:
        # Fallback: Recreate WS network (inform user this is being done)
        import networkx as nx
        N = 10000
        k = 8
        p = 0.1
        Gws = nx.watts_strogatz_graph(N, k, p, seed=42)
        Gws_csr = nx.to_scipy_sparse_array(Gws, format='csr')
        sparse.save_npz(network_output_path, Gws_csr)
        network_recreated = True
    else:
        network_recreated = False
else:
    network_recreated = False

# Load network as csr
G_csr = sparse.load_npz(network_output_path)
G_csr.shape, network_recreated
