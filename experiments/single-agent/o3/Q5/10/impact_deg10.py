
import numpy as np, json, os, scipy.sparse as sparse, networkx as nx, math
output_dir=os.path.join(os.getcwd(),'output')
G=sparse.load_npz(os.path.join(output_dir,'network.npz'))
G_nx = nx.from_scipy_sparse_array(G)

k_vals = np.array([d for n,d in G_nx.degree()])
N=len(k_vals)

de10_idx = np.where(k_vals==10)[0]
p10 = len(de10_idx)/N
mean_k = k_vals.mean()
second = (k_vals**2).mean()
q0 = (second - mean_k)/mean_k
print('orig',mean_k, second, q0)

# remove all deg10 nodes
keep = np.where(k_vals!=10)[0]
keep_deg=k_vals[keep]
mean_kp=keep_deg.mean()
second_p=(keep_deg**2).mean()
q1 = (second_p-mean_kp)/mean_kp
print('after removing all deg10', mean_kp, second_p, q1)

beta_over_gamma=1
R0p=beta_over_gamma*q1
print('R0 new', R0p)

