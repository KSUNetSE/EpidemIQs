
import numpy as np, networkx as nx, scipy.sparse as sparse, os, random, math, sys
from collections import Counter

# Parameters for network
N = 10000
p_deg10 = 0.1125
lambda_poisson = 2.0

rng = np.random.default_rng(seed=42)

degrees = []
for i in range(N):
    if rng.random() < p_deg10:
        degrees.append(10)
    else:
        deg = rng.poisson(lambda_poisson)
        degrees.append(deg)

# Ensure at least degree 0? we allow 0.

# Ensure the sum of degrees is even
if sum(degrees) % 2 == 1:
    degrees[0] += 1

# Build configuration model graph
G = nx.configuration_model(degrees, seed=42)
G = nx.Graph(G)  # remove parallel edges and self loops by making simple graph
G.remove_edges_from(nx.selfloop_edges(G))

# Recompute degrees after cleaning
k_values = [d for n, d in G.degree()]
mean_k = np.mean(k_values)
second_moment = np.mean(np.square(k_values))
q_val = (second_moment - mean_k) / mean_k

# Save network
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, 'network.npz')
csr = nx.to_scipy_sparse_array(G, format='csr')

sparse.save_npz(network_path, csr)

result = {
    'N': N,
    'edges': G.number_of_edges(),
    'mean_k': mean_k,
    'second_moment': second_moment,
    'q': q_val,
    'network_path': network_path
}
