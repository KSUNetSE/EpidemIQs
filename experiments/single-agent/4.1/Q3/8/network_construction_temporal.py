
import numpy as np
import scipy.sparse as sparse
import os
import networkx as nx

# Parameters for the temporal activity-driven network (ADN)
N = 1000  # number of nodes
alpha = 0.1  # activation probability
m = 2        # edges creatd per activation
T = 100
np.random.seed(42)

temporal_snapshots = []
for t in range(T):
    Gt = nx.Graph()
    Gt.add_nodes_from(range(N))
    active = np.random.rand(N) < alpha
    for i in np.where(active)[0]:
        neighbors = np.random.choice(np.delete(np.arange(N), i), size=m, replace=False)
        for j in neighbors:
            Gt.add_edge(i, j)
    temporal_snapshots.append(nx.to_scipy_sparse_array(Gt, format='csr'))

# Save temporal network as a list of adjacency matrices
os.makedirs('output', exist_ok=True)
for t, At in enumerate(temporal_snapshots):
    sparse.save_npz(os.path.join('output', f'temporal_network_{t}.npz'), At)

# Compute mean and second moment of degrees time-averaged
degree_matrix = np.zeros((N, T))
for t, Gt in enumerate(temporal_snapshots):
    G = nx.from_scipy_sparse_array(Gt)
    degrees = np.array([deg for _, deg in G.degree()])
    degree_matrix[:,t] = degrees

deg_mean_time = np.mean(degree_matrix)
deg2_mean_time = np.mean(degree_matrix**2)
# Save network info
with open(os.path.join('output','network_temporal_info.txt'),'w') as f:
    f.write(f'N={N}, alpha={alpha}, m={m}, T={T}\n')
    f.write('Temporal network as T=100 undirected edge snapshots for activity-driven dynamics.\n')
    f.write(f'Mean degree (time-averaged): {deg_mean_time:.2f}, Second moment <k^2> (time-averaged): {deg2_mean_time:.2f}\n')
