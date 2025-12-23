
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