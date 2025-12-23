
import numpy as np, networkx as nx, scipy.sparse as sparse, os, fastgemf as fg, pandas as pd, random, math, matplotlib

N=10000
# degree distribution: p10 fraction of nodes with degree 10, rest degree 2
p10=0.125  # approximate to match mean 3
num10=int(N*p10)
num2=N-num10
degree_sequence=[10]*num10+[2]*num2
random.shuffle(degree_sequence)

G=nx.configuration_model(degree_sequence, seed=42)
G=nx.Graph(G) # remove parallel edges and self-loops
G.remove_edges_from(nx.selfloop_edges(G))
print('Nodes', G.number_of_nodes(), 'Edges', G.number_of_edges())
# compute mean degree and q
k_list=[d for n,d in G.degree()]
mean_k=np.mean(k_list)
second_moment=np.mean([k**2 for k in k_list])
q=(second_moment-mean_k)/mean_k
print('mean k',mean_k,'q',q)

# save network
from scipy import sparse
csr=sparse.csr_matrix(nx.to_scipy_sparse_array(G))
output_dir=os.path.join(os.getcwd(),'output')
os.makedirs(output_dir, exist_ok=True)

sparse.save_npz(os.path.join(output_dir,'network.npz'), csr)
