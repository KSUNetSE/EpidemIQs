
# Code to construct both the temporal activity-driven network (for simulation steps) and the aggregated static network.
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import os
import matplotlib.pyplot as plt

n = 1000
alpha = 0.1  # probability of activation per node per time step
m = 2        # number of links created per active node
T = 500      # number of time steps over which to aggregate
np.random.seed(42)

# For each timestep, create a temporal contact list
edges_over_time = []
for t in range(T):
    active_nodes = np.where(np.random.rand(n) < alpha)[0]
    edges = set()
    for i in active_nodes:
        possible_targets = list(set(range(n)) - {i})
        targets = np.random.choice(possible_targets, size=m, replace=False)
        for j in targets:
            edge = tuple(sorted((i, j)))
            edges.add(edge)
    edges_over_time.append(list(edges))

# Build the time-aggregated static network with weighted edges (weight = frequency of contact)
aggr_edge_counts = {}
for edges in edges_over_time:
    for e in edges:
        aggr_edge_counts[e] = aggr_edge_counts.get(e, 0) + 1

G_static = nx.Graph()
G_static.add_nodes_from(range(n))
for (u, v), freq in aggr_edge_counts.items():
    G_static.add_edge(u, v, weight=freq)

# Save the static network adjacency (unweighted, for reference)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network_static.npz'), nx.to_scipy_sparse_array(G_static))

# Save the weighted adjacency matrix
weights = nx.to_scipy_sparse_array(G_static, weight='weight')
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network_static_weighted.npz'), weights)

# Network statistics for reporting
k_seq = [d for n, d in G_static.degree()]
k1 = np.mean(k_seq)
k2 = np.mean(np.square(k_seq))

# Plot degree distribution
deg = [G_static.degree(n) for n in G_static.nodes()]
plt.hist(deg, bins=30)
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution of Aggregated Static Network')
plt.savefig(os.path.join(os.getcwd(), 'output', 'degree-dist-static.png'))
plt.close()

# Save the edges_over_time for later temporal simulation
import pickle
with open(os.path.join(os.getcwd(), 'output', 'temporal_edges.pkl'), 'wb') as f:
    pickle.dump(edges_over_time, f)

return_vars = ['k1','k2','n']
