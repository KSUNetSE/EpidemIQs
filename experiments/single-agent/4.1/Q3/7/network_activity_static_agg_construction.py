
# Modeling phase: construct both networks for simulations
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import os
import matplotlib.pyplot as plt

current_directory = os.getcwd()
output_dir = os.path.join(current_directory, 'output')
os.makedirs(output_dir, exist_ok=True)

# Network parameters
N = 1000  # number of nodes
alpha = 0.1
m = 2
T = 100  # time window for aggregation

# Activity-driven temporal network is not a static network: at each time step, nodes randomly activate
# For the static, time-aggregated network, the weight of an edge represents the number of co-occurrences
# We'll construct the time-aggregated network

# For the static network, let's create an Erdos-Renyi with mean degree = mean_degree_instant = 0.4, but also show one with mean degree = mean_degree_agg = 40
mean_degree_instant = 2 * m * alpha
mean_degree_agg = 2 * m * alpha * T

# 1. Low connectivity network: mean degree = 0.4
p_low = mean_degree_instant / (N-1)
G_low = nx.erdos_renyi_graph(N, p_low)

# 2. Aggregated network: mean degree = 40
p_high = mean_degree_agg / (N-1)
G_agg = nx.erdos_renyi_graph(N, p_high)

# Save both networks
sparse.save_npz(os.path.join(output_dir, 'network_lowdeg.npz'), nx.to_scipy_sparse_array(G_low))
sparse.save_npz(os.path.join(output_dir, 'network_aggdeg.npz'), nx.to_scipy_sparse_array(G_agg))

# Plot degree distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
degs_low = [d for n, d in G_low.degree()]
degs_agg = [d for n, d in G_agg.degree()]
axes[0].hist(degs_low, bins=30)
axes[0].set_title('Degree dist, Mean deg=0.4 (instantaneous)')
axes[1].hist(degs_agg, bins=50)
axes[1].set_title('Degree dist, Mean deg=40 (aggregated)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'degree_dists.png'))
plt.close()

# Calculate moments
mean_degree_low = np.mean(degs_low)
mean_degree_agg_net = np.mean(degs_agg)
second_moment_low = np.mean(np.square(degs_low))
second_moment_agg = np.mean(np.square(degs_agg))

# Save details
return_vars = [
    'mean_degree_low', 'mean_degree_agg_net', 'second_moment_low', 'second_moment_agg'
]