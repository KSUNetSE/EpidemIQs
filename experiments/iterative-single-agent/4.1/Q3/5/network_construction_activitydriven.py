
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import sparse

np.random.seed(123)
# Network parameters
N = 1000
alpha = 0.1
m = 2
T = 1000  # Number of time steps for temporal network

# For the static aggregated network, we will build a weighted network where edge weights are the interaction frequency
edge_counts = {}

temporal_adj = np.zeros((N, N))

for t in range(T):
    # Each node, with probability alpha, becomes active and creates m random links
    active_nodes = np.where(np.random.rand(N) < alpha)[0]
    for node in active_nodes:
        targets = np.random.choice([i for i in range(N) if i != node], size=m, replace=False)
        for target in targets:
            # Save link (unordered for undirected)
            i, j = sorted([node, target])
            edge_counts[(i, j)] = edge_counts.get((i, j), 0) + 1
            temporal_adj[i, j] += 1
            temporal_adj[j, i] += 1  # undirected

# Build aggregated static weighted network 
G_agg = nx.Graph()
G_agg.add_nodes_from(range(N))
for (i, j), cnt in edge_counts.items():
    G_agg.add_edge(i, j, weight=cnt / T)

# Save the unweighted static aggregated network for SIR (for comparison with same mean degree as activity-driven)
G_static = nx.Graph()
G_static.add_nodes_from(range(N))
# To match the mean degree, select top-k frequent edges
# Each node forms alpha*m*T edges over T steps. Approx. mean degree: k_mean = 2*alpha*m
mean_degree_target = 2*alpha*m
num_edges = int(N * mean_degree_target / 2)

top_edges = sorted(edge_counts.items(), key=lambda x: -x[1])[:num_edges]
for (i, j), cnt in top_edges:
    G_static.add_edge(i, j)

# Save as sparse adjacency
os.makedirs('output', exist_ok=True)
sparse.save_npz(os.path.join('output', 'network-agg.npz'), nx.to_scipy_sparse_array(G_agg))
sparse.save_npz(os.path.join('output', 'network-unweighted.npz'), nx.to_scipy_sparse_array(G_static))

# Compute mean degree and second moment for both networks
mean_deg_agg = np.mean([d for (n, d) in G_agg.degree()])
k2_agg = np.mean([d**2 for (n, d) in G_agg.degree()])
mean_deg_static = np.mean([d for (n, d) in G_static.degree()])
k2_static = np.mean([d**2 for (n, d) in G_static.degree()])

# Degree distributions
plt.hist([d for n, d in G_agg.degree()], bins=20, alpha=0.6, label='Aggregated')
plt.hist([d for n, d in G_static.degree()], bins=20, alpha=0.6, label='Unweighted-Static')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.title('Degree Distributions')
plt.legend()
plt.savefig(os.path.join('output', 'degree_dist_aggregate_vs_static.png'))
plt.close()
