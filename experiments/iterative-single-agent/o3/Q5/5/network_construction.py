
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

import os, numpy as np, networkx as nx, random, scipy.sparse as sparse
import numpy as np, os
N=20000
# probabilities as derived
a=1/3 # degree 1
b=4/7 # degree 3
c=2/21 # degree 10
probs=[(1,a),(3,b),(10,c)]
# construct degree sequence
random.seed(42)
deg_seq=[]
for i in range(N):
    r=random.random()
    cumulative=0
    for k,p in probs:
        cumulative+=p
        if r<=cumulative:
            deg_seq.append(k)
            break
# ensure even sum
if sum(deg_seq)%2==1:
    deg_seq[0]+=1
G=nx.configuration_model(deg_seq, seed=42)
G=nx.Graph(G)
G.remove_edges_from(nx.selfloop_edges(G))
# save
output_dir=os.path.join(os.getcwd(),'output')
os.makedirs(output_dir,exist_ok=True)
csr=nx.to_scipy_sparse_array(G,format='csr')
from scipy import sparse as sp
sp.save_npz(os.path.join(output_dir,'network.npz'),csr)
# compute stats
k_vals=[d for n,d in G.degree()]
mean_k=float(np.mean(k_vals))
second_mom=float(np.mean(np.array(k_vals)**2))
q=(second_mom-mean_k)/mean_k
return_dict={'mean_k':mean_k,'second_moment':second_mom,'q':q,'num_nodes':N}

import os, numpy as np, networkx as nx, pandas as pd, scipy.sparse as sparse, random, math
import fastgemf as fg
from collections import Counter

N=10000
# generate degree sequence from NB(r=3,p=0.5)
r=3
p=0.5
# sample degrees
rng=np.random.default_rng(42)
max_deg=50
probs=[math.comb(k+r-1,k)*(1-p)**k * p**r for k in range(max_deg)]
probs=np.array(probs)
probs/=probs.sum()
# sample degrees until sum even
while True:
    degs=rng.choice(np.arange(len(probs)),size=N, p=probs)
    if degs.sum()%2==0: break
# build configuration model
G=nx.configuration_model(degs,seed=42)
G=nx.Graph(G) # remove parallel
G.remove_edges_from(nx.selfloop_edges(G))
print(nx.number_of_nodes(G), nx.number_of_edges(G))
# compute mean degree and q
k=np.array([d for n,d in G.degree()])
mean_k=k.mean()
q=(np.mean(k**2)-mean_k)/mean_k
print('mean k', mean_k, 'q', q)
# store network
from scipy import sparse
csr=nx.to_scipy_sparse_array(G,format='csr')
output_dir=os.path.join(os.getcwd(),'output')
os.makedirs(output_dir,exist_ok=True)
sparse.save_npz(os.path.join(output_dir,'network.npz'),csr)
