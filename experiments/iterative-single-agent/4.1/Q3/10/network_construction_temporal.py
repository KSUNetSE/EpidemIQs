
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

import numpy as np
import os

total_nodes = 1000
alpha = 0.1
m = 2
T = 1000
np.random.seed(42)

# Instead of trying to save the entire edge-lists-per-t as a single array, save using np.savez with variable-length lists
edge_lists_per_t = []
for t in range(T):
    active = np.random.rand(total_nodes) < alpha
    active_nodes = np.where(active)[0]
    edge_list_this_t = []
    for node in active_nodes:
        targets = np.random.choice([i for i in range(total_nodes) if i != node], size=m, replace=False)
        for target in targets:
            if node < target:
                edge_list_this_t.append((node, target))
            else:
                edge_list_this_t.append((target, node))
    edge_lists_per_t.append(edge_list_this_t)

# Save all edge lists per t as an object array using np.savez (each entry is a list)
temporal_network_path = os.path.join(os.getcwd(), "output", "temporal_network_edges.npz")
np.savez(temporal_network_path, *edge_lists_per_t)

# For visualization, show number of active nodes per timestep
active_counts = [len(edges)//2 for edges in edge_lists_per_t]  # each edge placed once between (i,j)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,3))
plt.hist(active_counts, bins=30, color='skyblue')
plt.xlabel('Number of activations per timestep')
plt.ylabel('Frequency')
plt.title('Distribution of Active Nodes per Time Step')
plt.tight_layout()
activity_pattern_path = os.path.join(os.getcwd(), "output", "temporal_activity_count_hist.png")
plt.savefig(activity_pattern_path)
plt.close()

(activity_pattern_path, temporal_network_path)