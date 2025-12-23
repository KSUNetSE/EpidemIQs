
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import os
import matplotlib.pyplot as plt

# Network parameters
N = 1000
k = 10 # mean degree
seed = 42
ws_p = 0.05 # rewiring probability for WS
k_side = k // 2 # Each node connected to k/2 neighbors on each side

# 1. Construct Watts-Strogatz network
G_ws = nx.watts_strogatz_graph(N, k, ws_p, seed=seed)

# Validate mean degree and second degree moment
degrees_ws = np.array([d for n, d in G_ws.degree()])
mean_k_ws = degrees_ws.mean()
mean_k2_ws = (degrees_ws**2).mean()

# Global clustering coefficient
clustering_ws = nx.transitivity(G_ws)

# Giant component assessment
gcc_ws = len(max(nx.connected_components(G_ws), key=len)) / N

# Degree distribution plot
plt.figure(figsize=(7,4))
plt.hist(degrees_ws, bins=np.arange(degrees_ws.min(), degrees_ws.max()+2)-0.5, color='navy', alpha=0.7)
plt.title('Degree Distribution, Watts-Strogatz N=1000, k=10, p=0.05')
plt.xlabel('Degree (k)')
plt.ylabel('#Nodes')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),"output","ws-degree-histogram.png"))
plt.close()

# Save network
ws_path = os.path.join(os.getcwd(),"output","network-wattsstrogatz.npz")
sparse.save_npz(ws_path, nx.to_scipy_sparse_array(G_ws))

return_vars = [
    'ws_path', 'mean_k_ws', 'mean_k2_ws', 'clustering_ws', 'gcc_ws', 'degrees_ws'
]
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import os
import matplotlib.pyplot as plt

# Network parameters
N = 1000
k = 10 # mean degree
seed = 42
# For G(n,p), p = k/(N-1)
er_p = k/(N-1)

# 2. Construct Erdős–Rényi random network
G_er = nx.erdos_renyi_graph(N, er_p, seed=seed)

degrees_er = np.array([d for n,d in G_er.degree()])
mean_k_er = degrees_er.mean()
mean_k2_er = (degrees_er**2).mean()

# Clustering coefficient
clustering_er = nx.transitivity(G_er)

# Giant component assessment
gcc_er = len(max(nx.connected_components(G_er), key=len)) / N

# Degree distribution plot
plt.figure(figsize=(7,4))
plt.hist(degrees_er, bins=np.arange(degrees_er.min(), degrees_er.max()+2)-0.5, color='tomato', alpha=0.7)
plt.title('Degree Distribution, Erdős–Rényi N=1000, k=10')
plt.xlabel('Degree (k)')
plt.ylabel('#Nodes')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),"output","er-degree-histogram.png"))
plt.close()

# Save network
er_path = os.path.join(os.getcwd(),"output","network-erdosrenyi.npz")
sparse.save_npz(er_path, nx.to_scipy_sparse_array(G_er))

return_vars = [
    'er_path', 'mean_k_er', 'mean_k2_er', 'clustering_er', 'gcc_er', 'degrees_er'
]