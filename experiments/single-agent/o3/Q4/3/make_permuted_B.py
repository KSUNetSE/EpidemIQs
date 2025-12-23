
import networkx as nx, numpy as np, scipy.sparse as sparse, os
output_dir=os.path.join(os.getcwd(),'output')
N=1000
GA=nx.erdos_renyi_graph(N,0.01,seed=42)
# B: Barabasi-Albert but permuted nodes
B_raw=nx.barabasi_albert_graph(N,3,seed=24)
perm=np.random.RandomState(123).permutation(N)
mapping={i:int(perm[i]) for i in range(N)}
GB=nx.relabel_nodes(B_raw,mapping)
# Save permuted B as network_B2
sparse.save_npz(os.path.join(output_dir,'network_B2.npz'), nx.to_scipy_sparse_array(GB))
