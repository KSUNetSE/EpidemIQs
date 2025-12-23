
import os, math, random, numpy as np, networkx as nx, scipy.sparse as sparse
from collections import Counter

# parameters
N=10000
mean_k=3.0
r=3.0 # negative bin k dispersion parameter
# generate degrees using neg bin (number of failures until r successes) distribution with mean mu=mean_k
mu=mean_k
p = r/(r+mu)

degrees = np.random.negative_binomial(r, p, size=N)
# fix odd sum to make even
if degrees.sum()%2==1:
    idx=random.randrange(N)
    degrees[idx] +=1
# build configuration model simple graph
G_multi = nx.configuration_model(degrees, create_using=nx.MultiGraph())
G = nx.Graph()
for u,v in G_multi.edges():
    if u!=v:
        G.add_edge(u,v)

print('Nodes',G.number_of_nodes(),'Edges',G.number_of_edges())

# compute moments
k = np.array([deg for _,deg in G.degree()])
mean_k_emp = k.mean()
second_moment = (k*k).mean()
q_emp = (second_moment - mean_k_emp)/mean_k_emp
print('mean',mean_k_emp,'second',second_moment,'q',q_emp)

# save network
import scipy.sparse as sparse, numpy as np, os
adj = nx.to_scipy_sparse_array(G, dtype=np.uint8)
path = os.path.join(os.getcwd(),'output','network.npz')
sparse.save_npz(path, adj)
print('saved network',path)
