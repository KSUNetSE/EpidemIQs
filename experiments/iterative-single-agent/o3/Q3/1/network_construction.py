
import networkx as nx, numpy as np, os, scipy.sparse as sparse, math
import random, json, pathlib, itertools
from collections import Counter

output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

n=1000
# Layer A: Barabasi-Albert with m=3
G_A = nx.barabasi_albert_graph(n, 3, seed=42)
# Layer B: create another BA but with node relabeling to reduce correlation
G_B_raw = nx.barabasi_albert_graph(n, 3, seed=99)
# relabel nodes randomly
perm = list(range(n))
random.seed(123)
random.shuffle(perm)
map_dict = {old: new for old, new in zip(range(n), perm)}
G_B = nx.relabel_nodes(G_B_raw, map_dict)

# compute metrics
ks_A = np.array([d for _,d in G_A.degree()])
ks_B = np.array([d for _,d in G_B.degree()])
mean_k_A = ks_A.mean()
mean_k_B = ks_B.mean()
mean_k2_A = np.mean(ks_A**2)
mean_k2_B = np.mean(ks_B**2)

# compute overlap of top degree nodes (central nodes)
central_A = set(sorted(G_A.degree, key=lambda x: x[1], reverse=True)[:50])
central_A = set([node for node,_ in central_A])
central_B = set(sorted(G_B.degree, key=lambda x: x[1], reverse=True)[:50])
central_B = set([node for node,_ in central_B])
central_overlap = len(central_A.intersection(central_B))/50.0

# spectral radius (largest eigenvalue) approximate using numpy for adjacency matrix
import scipy.sparse.linalg as spla
lam_A = spla.eigs(nx.to_scipy_sparse_array(G_A), k=1, which='LR', return_eigenvectors=False)[0].real
lam_B = spla.eigs(nx.to_scipy_sparse_array(G_B), k=1, which='LR', return_eigenvectors=False)[0].real

# Save networks
sparse.save_npz(os.path.join(output_dir, 'networkA.npz'), nx.to_scipy_sparse_array(G_A))
sparse.save_npz(os.path.join(output_dir, 'networkB.npz'), nx.to_scipy_sparse_array(G_B))

result = {'mean_k_A':mean_k_A,'mean_k2_A':mean_k2_A,'mean_k_B':mean_k_B,'mean_k2_B':mean_k2_B,'central_overlap':central_overlap,'lambda_A':lam_A,'lambda_B':lam_B}

result
import networkx as nx, numpy as np, scipy.sparse as sparse, os
n=1000
# Layer A Barabasi-Albert
G_A=nx.barabasi_albert_graph(n,3,seed=42)
# Layer B Watts-Strogatz
G_B=nx.watts_strogatz_graph(n,6,0.1,seed=24)
# Compute degree moments
kA=np.array([d for _,d in G_A.degree()])
mean_kA=kA.mean(); mean_k2A=(kA**2).mean()
kB=np.array([d for _,d in G_B.degree()])
mean_kB=kB.mean(); mean_k2B=(kB**2).mean()
# Largest eigenvalues
from scipy.sparse.linalg import eigs
lamA=eigs(nx.to_scipy_sparse_array(G_A),k=1,which='LR',return_eigenvectors=False)[0].real
lamB=eigs(nx.to_scipy_sparse_array(G_B),k=1,which='LR',return_eigenvectors=False)[0].real
out_dir=os.path.join(os.getcwd(),'output')
os.makedirs(out_dir,exist_ok=True)
sparse.save_npz(os.path.join(out_dir,'layerA_network.npz'), nx.to_scipy_sparse_array(G_A))
sparse.save_npz(os.path.join(out_dir,'layerB_network.npz'), nx.to_scipy_sparse_array(G_B))
print({'mean_kA':mean_kA,'mean_k2A':mean_k2A,'lambdaA':lamA,'mean_kB':mean_kB,'mean_k2B':mean_k2B,'lambdaB':lamB})