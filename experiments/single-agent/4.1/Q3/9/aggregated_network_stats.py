
import numpy as np
n = 1000
alpha = 0.1
m = 2
num_timesteps = 1000
adjacency_time = []

for t in range(num_timesteps):
    adj = np.zeros((n, n))
    for i in range(n):
        if np.random.rand() < alpha:
            partners = np.random.choice(np.delete(np.arange(n), i), m, replace=False)
            adj[i, partners] = 1
            adj[partners, i] = 1  # undirected
    adjacency_time.append(adj)
# Aggregate the adjacency matrices
total_agg = np.sum(adjacency_time, axis=0)
static_edge_weights = total_agg
# Calculate mean degree and degree squared for static aggregated network
degrees = np.sum(total_agg > 0, axis=1)
k_avg = np.mean(degrees)
k2_avg = np.mean(degrees**2)
k_avg_weight = np.mean(np.sum(static_edge_weights, axis=1))
k2_avg_weight = np.mean(np.sum(static_edge_weights, axis=1)**2)

return_vars = ["k_avg", "k2_avg", "k_avg_weight", "k2_avg_weight", "total_agg"]