
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

import networkx as nx
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import os

# Parameters
total_nodes = 1000
alpha = 0.1  # Probability of activation per time step
m = 2        # Number of links formed per activation
T = 1000     # Number of time steps for temporal aggregation (long enough for good statistics)
np.random.seed(42)  # for reproducibility

# Each node will record its contacts for static aggregation
contact_counts = {}
for node in range(total_nodes):
    contact_counts[node] = {}

# Simulate the activity-driven temporal network, aggregating contacts
for t in range(T):
    active = np.random.rand(total_nodes) < alpha
    active_nodes = np.where(active)[0]
    np.random.shuffle(active_nodes)
    for node in active_nodes:
        # Pick m targets uniformly at random (excluding self), potentially repeated
        targets = np.random.choice([i for i in range(total_nodes) if i != node], size=m, replace=False)
        for target in targets:
            # Record in contact_counts (order doesn't matter for undirected networks)
            i, j = min(node, target), max(node, target)
            contact_counts[i][j] = contact_counts[i].get(j, 0) + 1

# Build the aggregated weighted network from contact_counts
G_static = nx.Graph()
for i in contact_counts:
    for j in contact_counts[i]:
        weight = contact_counts[i][j] / T  # frequency per time unit
        G_static.add_edge(i, j, weight=weight)

# Save the weighted adjacency matrix as sparse matrix (non-weighted for now, weights as attribute for reference)
A = nx.to_scipy_sparse_array(G_static, nodelist=range(total_nodes), weight=None)
sparse.save_npz(os.path.join(os.getcwd(), "output", "network_static.npz"), A)

# Compute degree metrics
k = np.array([d for n, d in G_static.degree()])
k_mean = np.mean(k)
k2_mean = np.mean(k ** 2)

# Plot degree distribution of aggregated static network
plt.figure(figsize=(6,4))
plt.hist(k, bins=30)
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Degree Distribution (Aggregated Static Network)')
plt.tight_layout()
degree_dist_path = os.path.join(os.getcwd(), "output", "static_degree_dist.png")
plt.savefig(degree_dist_path)
plt.close()

# Save for future analysis
network_path = os.path.join(os.getcwd(), "output", "network_static.npz")
plot_paths = {degree_dist_path: "Degree distribution of the time-aggregated static network."}

# Return metrics needed for SIR parameterization as well
(k_mean, k2_mean, network_path, plot_paths)