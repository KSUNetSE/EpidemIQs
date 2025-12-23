
import os, json, networkx as nx, numpy as np, scipy.sparse as sparse
from collections import Counter
np.random.seed(42)
N = 10000
prop_high = 0.125  # 12.5% high degree 10
n_high = int(N*prop_high)
# degree list: high degree 10, rest degree 2
deg_list = [10]*n_high + [2]*(N-n_high)
# ensure sum of degrees even
if sum(deg_list)%2==1:
    deg_list[-1]+=1
G = nx.configuration_model(deg_list, create_using=nx.Graph())
G.remove_edges_from(nx.selfloop_edges(G))
# compute degree moments
k = np.array([d for _,d in G.degree()])
mean_k = k.mean()
mean_k2 = (k**2).mean()
q = (mean_k2 - mean_k)/mean_k
mean_k, mean_k2, q