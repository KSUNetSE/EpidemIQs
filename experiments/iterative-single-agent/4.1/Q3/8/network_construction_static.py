
import networkx as nx
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters for the static network aggregation of an activity-driven network (ADN)
N = 1000  # number of nodes
alpha = 0.1  # activation probability per node per time step
m = 2        # number of edges created upon activation
T = 100  # number of time steps to aggregate over
np.random.seed(42)

# Step 1: Simulate ADN and Keep Edge Counts
edge_counts = {}
for t in range(T):
    active = np.random.rand(N) < alpha
    for i in np.where(active)[0]:
        neighbors = np.random.choice(np.delete(np.arange(N), i), size=m, replace=False)
        for j in neighbors:
            e = tuple(sorted((i, j)))
            edge_counts[e] = edge_counts.get(e, 0) + 1

# Step 2: Create static, weighted aggregated network
G_static = nx.Graph()
G_static.add_nodes_from(range(N))
for (i, j), w in edge_counts.items():
    G_static.add_edge(i, j, weight=w)

# Step 3: Save as sparse matrix for compatibility with downstream simulation
A_static = nx.to_scipy_sparse_array(G_static, weight='weight', format='csr')
os.makedirs('output', exist_ok=True)
sparse.save_npz(os.path.join('output','network_static_agg.npz'), A_static)

# Step 4: Plot degree distribution for static aggregated network
degrees = [d for n, d in G_static.degree()]
plt.figure()
plt.hist(degrees, bins=20, color='lightblue', edgecolor='black')
plt.title('Degree Distribution of Aggregated Static Network')
plt.xlabel('Degree')
plt.ylabel('Node Count')
plt.savefig(os.path.join('output','degdist_static_agg.png'))

# Step 5: Calculate mean degree and degree second moment
k_mean = np.mean(degrees)
k2_mean = np.mean(np.square(degrees))

# Step 6: Save reasoning/logical info
with open(os.path.join('output','network_static_agg_info.txt'),'w') as f:
    f.write(f'N={N}, alpha={alpha}, m={m}, T={T}\n')
    f.write(f'Aggregated weighted static network simulated from ADN temporal process.\n')
    f.write(f'Mean degree <k>: {k_mean:.2f}, Second moment <k^2>: {k2_mean:.2f}\n')
