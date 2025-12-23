
import os, scipy.sparse as sparse, networkx as nx, numpy as np
output_dir = os.path.join(os.getcwd(),'output')
os.makedirs(output_dir, exist_ok=True)
# Regenerate same networks as earlier
N=1000
GA=nx.erdos_renyi_graph(N,0.01,seed=42)
GB=nx.barabasi_albert_graph(N,3,seed=24)
# Save
sparse.save_npz(os.path.join(output_dir,'network_A.npz'), nx.to_scipy_sparse_array(GA))
sparse.save_npz(os.path.join(output_dir,'network_B.npz'), nx.to_scipy_sparse_array(GB))
