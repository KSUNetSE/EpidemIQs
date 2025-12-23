
import json, os
import numpy as np
import scipy.sparse as sparse
import networkx as nx

output_dir=os.path.join(os.getcwd(),'output')
G=sparse.load_npz(os.path.join(output_dir,'network.npz'))
G_nx = nx.from_scipy_sparse_array(G)
k_vals = np.array([d for n,d in G_nx.degree()])
mean_k = k_vals.mean()
second = (k_vals**2).mean()
q = (second-mean_k)/mean_k
print('mean',mean_k,'q',q)

deg10 = np.where(k_vals==10)[0]
r_k = np.ones(len(k_vals), dtype=bool)
# simulate removal of deg10 nodes
keep_indices = np.where(k_vals!=10)[0]
keep_deg = k_vals[keep_indices]
mean_kp = keep_deg.mean()
second_p = (keep_deg**2).mean()
q_p = (second_p - mean_kp)/mean_kp
print('after removal, mean', mean_kp, 'q', q_p)