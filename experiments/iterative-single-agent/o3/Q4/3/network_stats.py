
import random, os, math, scipy.sparse as sparse, networkx as nx, numpy as np, json, textwrap, pandas as pd
N=20000
prob = [(1,0.333333), (3,0.571429), (10,0.095238)]
# generate degree sequence
degrees=[None]*N
for i in range(N):
    r=random.random()
    cumulative=0.0
    for k,p in prob:
        cumulative+=p
        if r<=cumulative:
            degrees[i]=k
            break
# ensure even sum
if sum(degrees)%2==1:
    degrees[0]+=1
# build config model
G=nx.configuration_model(degrees, seed=42)
G=nx.Graph(G) # remove parallel edges
G.remove_edges_from(nx.selfloop_edges(G))
# compute stats
k_vals=[d for n,d in G.degree()]
mean_k=np.mean(k_vals)
second_moment=np.mean(np.array(k_vals)**2)
q=(second_moment-mean_k)/mean_k
mean_k, second_moment, q
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import scipy.linalg as la
import os, json, math, sys, random
N=1000
# layer A: Erdos Renyi
p=0.01
GA=nx.erdos_renyi_graph(N,p,seed=42)
# layer B: Barabasi-Albert
m=3
GB=nx.barabasi_albert_graph(N,m,seed=24)
# Compute mean degree and second moment
kA=np.array([d for n,d in GA.degree()])
mean_kA=kA.mean()
second_moment_A=(kA**2).mean()

kB=np.array([d for n,d in GB.degree()])
mean_kB=kB.mean()
second_moment_B=(kB**2).mean()

# spectral radius (largest eigenvalue) of adjacency matrices
import scipy.sparse.linalg as spla
A_csr=nx.to_scipy_sparse_array(GA,format='csr')
B_csr=nx.to_scipy_sparse_array(GB,format='csr')
lamA=spla.eigs(A_csr,k=1,which='LR',return_eigenvectors=False)[0].real
lamB=spla.eigs(B_csr,k=1,which='LR',return_eigenvectors=False)[0].real
print(mean_kA, second_moment_A, lamA)
print(mean_kB, second_moment_B, lamB)
