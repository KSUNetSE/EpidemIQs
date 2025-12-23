
import numpy as np, networkx as nx, scipy.sparse as sparse, os, random, math, sys
from collections import Counter

# Parameters for network
N = 10000
p_deg10 = 0.1125
lambda_poisson = 2.0

rng = np.random.default_rng(seed=42)

degrees = []
for i in range(N):
    if rng.random() < p_deg10:
        degrees.append(10)
    else:
        deg = rng.poisson(lambda_poisson)
        degrees.append(deg)

# Ensure at least degree 0? we allow 0.

# Ensure the sum of degrees is even
if sum(degrees) % 2 == 1:
    degrees[0] += 1

# Build configuration model graph
G = nx.configuration_model(degrees, seed=42)
G = nx.Graph(G)  # remove parallel edges and self loops by making simple graph
G.remove_edges_from(nx.selfloop_edges(G))

# Recompute degrees after cleaning
k_values = [d for n, d in G.degree()]
mean_k = np.mean(k_values)
second_moment = np.mean(np.square(k_values))
q_val = (second_moment - mean_k) / mean_k

# Save network
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, 'network.npz')
csr = nx.to_scipy_sparse_array(G, format='csr')

sparse.save_npz(network_path, csr)

result = {
    'N': N,
    'edges': G.number_of_edges(),
    'mean_k': mean_k,
    'second_moment': second_moment,
    'q': q_val,
    'network_path': network_path
}

import numpy as np, networkx as nx, scipy.sparse as sparse, os, random, math
N=10000
r=3
p=0.5
# generate NB degrees up to maybe high tail
# Use numpy.random.negative_binomial but note mean = r*(1-p)/p
mean=r*(1-p)/p
print('mean',mean)
# generate degrees, then adjust to even sum
while True:
    degs=np.random.negative_binomial(r, p, N)
    if degs.sum()%2==0 and degs.mean()>0:
        break
print('generated mean',degs.mean(), 'var',degs.var())
# compute q
k=degs
mean_k=k.mean()
mean_k2=np.mean(k*(k-1))
q=mean_k2/mean_k
print('q',q)
# Save network
G=nx.configuration_model(k, create_using=nx.Graph())
G.remove_edges_from(nx.selfloop_edges(G))
print('n',G.number_of_nodes(),'m',G.number_of_edges())
# Save adjacency matrix
csr=nx.to_scipy_sparse_array(G, format='csr')
import scipy.sparse as sp
os.makedirs(os.path.join(os.getcwd(),'output'), exist_ok=True)
sp.save_npz(os.path.join(os.getcwd(),'output','network.npz'), csr)
