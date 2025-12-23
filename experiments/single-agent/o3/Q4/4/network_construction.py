
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

import numpy as np, networkx as nx, os, scipy.sparse as sparse, math, random
N=20000
r=3
p=0.5
# sample degrees from NB until even sum
while True:
    degrees=np.random.negative_binomial(r, p, size=N)
    # ensure at least degree 1?
    # Keep as is; can include zeros
    if degrees.sum()%2==0 and degrees.mean()>0:
        break
print('mean', degrees.mean())
print('second moment', np.mean(degrees**2))
# Build configuration model
G=nx.configuration_model(degrees, create_using=None, seed=42)
G=nx.Graph(G)  # simple
G.remove_edges_from(nx.selfloop_edges(G))
print('nodes', G.number_of_nodes(), 'edges', G.number_of_edges())
# compute metrics
k_arr=np.array([d for n,d in G.degree()])
mean_k=k_arr.mean()
second_moment=np.mean(k_arr**2)
q=(second_moment - mean_k)/mean_k
print('mean_k',mean_k,'second_moment',second_moment,'q',q)
# save network
out_dir=os.path.join(os.getcwd(),'output')
import pathlib; pathlib.Path(out_dir).mkdir(exist_ok=True)
net_path=os.path.join(out_dir,'network.npz')
sparse.save_npz(net_path, nx.to_scipy_sparse_array(G))
print('saved', net_path)

import os, json, random, math, numpy as np, networkx as nx, scipy.sparse as sparse, pathlib, sys
from collections import Counter

N = 10000
p_deg10 = 0.12 # 12% nodes with degree 10
n_deg10 = int(N * p_deg10)
n_other = N - n_deg10
# degrees for other nodes from Poisson with lambda=2 but truncated at 9
lam = 2.0
others_degrees = np.random.poisson(lam, size=n_other)
others_degrees = np.clip(others_degrees, 0, 9)
# ensure sum of stubs is even (including deg10 nodes)
deg_seq = list(others_degrees) + [10]*n_deg10
# adjust parity
if sum(deg_seq) % 2 == 1:
    # increment degree of a random non-9 other node by 1
    for i in range(len(deg_seq)):
        if deg_seq[i] < 9:
            deg_seq[i] += 1
            break
# create configuration model (multigraph) then convert to simple graph
G_multi = nx.configuration_model(deg_seq, seed=42)
G = nx.Graph(G_multi)  # remove parallel edges
G.remove_edges_from(nx.selfloop_edges(G))
# compute mean degree and second moment
k_vals = np.array([d for n, d in G.degree()])
mean_k = k_vals.mean()
second_moment = (k_vals**2).mean()
q = (second_moment - mean_k)/mean_k
result = {
    'N': N,
    'mean_k': mean_k,
    'second_moment': second_moment,
    'q': q,
    'degree_hist': Counter(k_vals)
}
# save network
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
net_path = os.path.join(output_dir, 'network.npz')
sparse.save_npz(net_path, nx.to_scipy_sparse_array(G))
result_path = os.path.join(output_dir,'network_stats.json')
with open(result_path,'w') as f:
    json.dump({'mean_k':mean_k,'second_moment':second_moment,'q':q}, f)

return_vars = ['result', 'net_path']
import networkx as nx
import numpy as np
import os
import scipy.sparse as sparse
from numpy.linalg import eigvals
import matplotlib.pyplot as plt

output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

n = 1000
# Layer A: Barabasi-Albert scale-free
m = 3
G_A = nx.barabasi_albert_graph(n, m, seed=42)
# Layer B1: Identical to A (correlated) for model 1
G_B1 = G_A.copy()
# Layer B2: Erdős-Rényi random with same mean degree <k> ~ 2m
p = (2*m)/ (n-1)
G_B2 = nx.erdos_renyi_graph(n, p, seed=123)

# Save networks
sparse.save_npz(os.path.join(output_dir, 'layerA.npz'), nx.to_scipy_sparse_array(G_A))
sparse.save_npz(os.path.join(output_dir, 'layerB1.npz'), nx.to_scipy_sparse_array(G_B1))
sparse.save_npz(os.path.join(output_dir, 'layerB2.npz'), nx.to_scipy_sparse_array(G_B2))

# Compute statistics
k_A = np.array([d for _, d in G_A.degree()])
avg_k_A = k_A.mean()
second_moment_A = (k_A**2).mean()
# largest eigenvalue
lambda1_A = max(eigvals(nx.to_numpy_array(G_A))).real

k_B1 = k_A  # identical
avg_k_B1 = avg_k_A
second_moment_B1 = second_moment_A
lambda1_B1 = lambda1_A

k_B2 = np.array([d for _, d in G_B2.degree()])
avg_k_B2 = k_B2.mean()
second_moment_B2 = (k_B2**2).mean()
lambda1_B2 = max(eigvals(nx.to_numpy_array(G_B2))).real

network_stats = {
    'avg_k_A': avg_k_A,
    'second_moment_A': second_moment_A,
    'lambda1_A': lambda1_A,
    'avg_k_B1': avg_k_B1,
    'second_moment_B1': second_moment_B1,
    'lambda1_B1': lambda1_B1,
    'avg_k_B2': avg_k_B2,
    'second_moment_B2': second_moment_B2,
    'lambda1_B2': lambda1_B2
}

network_stats
import os, networkx as nx, numpy as np, scipy.sparse as sparse, math
from scipy.sparse.linalg import eigs

os.makedirs(os.path.join(os.getcwd(),'output'), exist_ok=True)

N=1000
m=3
G_A=nx.barabasi_albert_graph(N,m,seed=42)
p=0.01
G_B=nx.erdos_renyi_graph(N,p,seed=24)

sparse.save_npz(os.path.join(os.getcwd(),'output','layerA.npz'), nx.to_scipy_sparse_array(G_A))
sparse.save_npz(os.path.join(os.getcwd(),'output','layerB.npz'), nx.to_scipy_sparse_array(G_B))

A_A=nx.to_scipy_sparse_array(G_A, dtype=float)
A_B=nx.to_scipy_sparse_array(G_B, dtype=float)

lamA=eigs(A_A,k=1,which='LR',return_eigenvectors=False)[0].real
lamB=eigs(A_B,k=1,which='LR',return_eigenvectors=False)[0].real

kA=np.array([d for _,d in G_A.degree()])
mean_kA=kA.mean(); mean_k2A=(kA**2).mean()

kB=np.array([d for _,d in G_B.degree()])
mean_kB=kB.mean(); mean_k2B=(kB**2).mean()

result={'lamA':lamA,'lamB':lamB,'mean_kA':mean_kA,'mean_k2A':mean_k2A,'mean_kB':mean_kB,'mean_k2B':mean_k2B}

import networkx as nx, numpy as np, scipy.sparse as sparse, os, json, math

# parameters
N=500
p=0.03
layerA=nx.fast_gnp_random_graph(N,p,seed=42,directed=False)
layerB=nx.barabasi_albert_graph(N,3,seed=24)

# compute mean degree and second moment for each layer
kA=np.array([d for n,d in layerA.degree()])
mean_kA=kA.mean()
sec_kA=np.mean(kA**2)

kB=np.array([d for n,d in layerB.degree()])
mean_kB=kB.mean()
sec_kB=np.mean(kB**2)

# spectral radius (largest eigenvalue) of adjacency matrices
import numpy.linalg as LA
lamA=LA.eigvals(nx.to_numpy_array(layerA)).real.max()
lamB=LA.eigvals(nx.to_numpy_array(layerB)).real.max()

beta1=1.5/lamA # with delta1=1
beta2=1.5/lamB # delta2=1

data={'N':N,'mean_kA':mean_kA,'sec_kA':sec_kA,'mean_kB':mean_kB,'sec_kB':sec_kB,'lambdaA':lamA,'lambdaB':lamB,'beta1':beta1,'beta2':beta2}
print(data)

# save networks
os.makedirs(os.path.join(os.getcwd(),'output'),exist_ok=True)
sparse.save_npz(os.path.join(os.getcwd(),'output','layerA.npz'), nx.to_scipy_sparse_array(layerA))
sparse.save_npz(os.path.join(os.getcwd(),'output','layerB.npz'), nx.to_scipy_sparse_array(layerB))