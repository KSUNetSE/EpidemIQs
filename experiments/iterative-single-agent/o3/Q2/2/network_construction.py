
import os, networkx as nx, numpy as np, scipy.sparse as sparse, matplotlib.pyplot as plt
import json, math, pandas as pd, random, itertools
from collections import Counter

# Create output dir
os.makedirs(os.path.join(os.getcwd(), 'output'), exist_ok=True)

# Parameters for network
N = 1000
p = 0.01  # ER with mean degree ~10 (N*p)
G = nx.erdos_renyi_graph(N, p, seed=42)
# Ensure giant component, assume fine

# Save network
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network.npz'), nx.to_scipy_sparse_array(G))

# Compute moments
degrees = np.array([d for n, d in G.degree()])
mean_k = degrees.mean()
second_moment = (degrees**2).mean()

mean_k, second_moment
import os, networkx as nx, numpy as np, scipy.sparse as sp, json
from datetime import datetime
N=1000
mean_k=10
p=mean_k/(N-1)
G=nx.erdos_renyi_graph(N,p,seed=42)
# Ensure connected? Not necessary.
k=np.array([d for n,d in G.degree()])
mean_deg=k.mean()
second_moment=(k**2).mean()
output_dir=os.path.join(os.getcwd(),'output')
os.makedirs(output_dir,exist_ok=True)
sp.save_npz(os.path.join(output_dir,'network.npz'), nx.to_scipy_sparse_array(G))
# Save some details
details={'N':N,'mean_k':mean_deg,'k2':second_moment,'timestamp':str(datetime.utcnow())}
with open(os.path.join(output_dir,'network_details.json'),'w') as f:
    json.dump(details,f)
print(details)