
import networkx as nx, numpy as np, random, math, os
N=20000
z=3
# Generate Poisson degree sequence
seq=np.random.poisson(z, N)
# ensure even sum
if sum(seq)%2==1:
    seq[0]+=1
G=nx.configuration_model(seq, create_using=nx.Graph())
G.remove_edges_from(nx.selfloop_edges(G))
print('Generated',G.number_of_nodes(),'nodes',G.number_of_edges(),'edges')