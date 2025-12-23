
import os, json, numpy as np, networkx as nx, scipy.sparse as sparse
from math import comb
# Create output directory
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

n = 10000  # number of nodes
r = 3  # NB parameter r
p = 0.5  # NB parameter p to get mean 3, var 6
# generate degree sequence
rng = np.random.default_rng(seed=42)
deg_seq = rng.negative_binomial(r, p, size=n)
# Ensure even sum of degrees (configuration model requirement)
if sum(deg_seq) % 2 == 1:
    idx = rng.integers(0, n)
    deg_seq[idx] += 1

G = nx.configuration_model(deg_seq, seed=42)
G = nx.Graph(G)  # remove parallel edges self loops later
G.remove_edges_from(nx.selfloop_edges(G))

# compute degree moments
k = np.array([d for _, d in G.degree()])
mean_k = k.mean()
mean_k2 = (k**2).mean()
q_emp = (mean_k2 - mean_k) / mean_k

# Save network
sparse.save_npz(os.path.join(output_dir, 'network.npz'), nx.to_scipy_sparse_array(G))

result = {
    'n_nodes': n,
    'mean_k': mean_k,
    'mean_k2': mean_k2,
    'q_empirical': q_emp,
    'deg10_fraction': (k==10).sum()/n
}

json.dumps(result)
import networkx as nx, random, os, scipy.sparse as sparse
import numpy as np
n=10000
x=0.1091
y2=0.7637
y3=0.1272
num_k10=int(round(x*n))
remaining=n-num_k10
num_k2=int(round(y2*n))
num_k3=n-num_k10-num_k2
# Build degree list
degree_sequence=[10]*num_k10 + [2]*num_k2 + [3]*num_k3
random.shuffle(degree_sequence)
print(len(degree_sequence), sum(degree_sequence)/n)
G=nx.configuration_model(degree_sequence, seed=42)
# convert to simple
G=nx.Graph(G)  # removes parallel edges
G.remove_edges_from(nx.selfloop_edges(G))
print(G.number_of_nodes(), G.number_of_edges())
# save network
import scipy.sparse as sp
csr=nx.to_scipy_sparse_array(G, format='csr')
output_dir=os.path.join(os.getcwd(),'output')
os.makedirs(output_dir, exist_ok=True)
sp.save_npz(os.path.join(output_dir,'network.npz'), csr)
# compute mean degree moments
k=np.array(csr.sum(axis=1)).flatten()
mean_k=k.mean()
mean_k2=(k**2).mean()
q=(mean_k2-mean_k)/mean_k
print('mean',mean_k,'k2',mean_k2,'q',q)
