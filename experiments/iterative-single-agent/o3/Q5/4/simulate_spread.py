
import os, random, networkx as nx, numpy as np, scipy.sparse as sp, pandas as pd, math, pickle
output_dir=os.path.join(os.getcwd(),'output')
# load network
csr=sp.load_npz(os.path.join(output_dir,'network.npz'))
G=nx.from_scipy_sparse_array(csr)
N=G.number_of_nodes()
print('N',N)
# scenario 0: baseline
# Infect single random node, SI spread with prob 1 (so entire component infected). We'll record outbreak size.

def outbreak_size(G):
    start=random.choice(list(G.nodes()))
    comp=len(nx.node_connected_component(G,start))
    return comp

def run_trials(G, trials=100):
    sizes=[outbreak_size(G) for _ in range(trials)]
    return np.mean(sizes), np.percentile(sizes, [50, 90])

# baseline sizes=might be large
mean0, med0=run_trials(G)
print('baseline mean',mean0)

# scenario random vaccination 75%
import itertools
f=0.75
nodes=list(G.nodes())
random.seed(1)
remove_nodes=set(random.sample(nodes,int(f*N)))
G_rand=G.copy()
G_rand.remove_nodes_from(remove_nodes)
mean_rand, med_rand=run_trials(G_rand)
print('rand vacc mean',mean_rand)

# scenario targeted degree10 (all degree 10 nodes)
remove_target=[n for n,d in G.degree() if d==10]
G_tar=G.copy()
G_tar.remove_nodes_from(remove_target)
mean_tar, med_tar=run_trials(G_tar)
print('target vacc mean',mean_tar)
