
import numpy as np
import networkx as nx
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import os

# Parameters
N = 1000  # number of nodes
alpha = 0.1  # activation probability per time step
m = 2  # connections per activation
T = 500  # number of time steps to aggregate

# Storage for temporal edges
edge_lists = []
weighted_edge_dict = {}

# Simulate the activity-driven temporal network
for t in range(T):
    active = np.random.rand(N) < alpha  # Boolean: which nodes are active this timestep?
    active_nodes = np.where(active)[0]
    new_edges = []
    for src in active_nodes:
        targets = np.random.choice(np.delete(np.arange(N), src), m, replace=False)
        for dst in targets:
            edge = tuple(sorted([src, dst]))
            new_edges.append(edge)
            # For aggregation
            if edge in weighted_edge_dict:
                weighted_edge_dict[edge] += 1
            else:
                weighted_edge_dict[edge] = 1
    edge_lists.append(new_edges)

# Build the aggregate (time-collapsed) network w/ weights
G_agg = nx.Graph()
for (u, v), w in weighted_edge_dict.items():
    G_agg.add_edge(u, v, weight=w)

# Save aggregate network adjacency (unweighted for SIR, but retain edge weight info in G_agg)
sparse.save_npz(os.path.join(os.getcwd(), "output", "network_aggregate.npz"), nx.to_scipy_sparse_array(G_agg))

# Save degree distribution plot of the aggregate network
plt.figure(figsize=(7,4))
deg = [G_agg.degree(n) for n in G_agg.nodes()]
plt.hist(deg, bins=np.arange(max(deg)+2)-0.5, color='c', alpha=0.8)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree Distribution of the Aggregated Network")
plt.savefig(os.path.join(os.getcwd(), "output", "degree_dist_aggregate.png"))
plt.close()

# Also store degree stats for param calculation
k_mean = np.mean(deg)
k2_mean = np.mean(np.array(deg)**2)
