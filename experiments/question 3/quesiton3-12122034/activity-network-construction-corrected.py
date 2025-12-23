
import numpy as np
import networkx as nx
from scipy import sparse
import os
import matplotlib.pyplot as plt

# Re-calculation of correct observation window for target <k>
N = 1000
alpha = 0.1
m = 2
desired_mean_degree = 2 * m * alpha  # Should be 0.4
num_nodes = N

# Let aggregation period be delta_T.
# In T steps, expected total links: alpha*N*T*m (since alpha*N*T activations, m partners per activation)
# Each link is among N*(N-1)/2 potential unique undirected pairs.
# Expected average degree in aggregated static network after T steps:
# <k> = 2*L/N where L=number of unique undirected edges present
# But with repeated sampling, number of unique edges will be less than number of edge activations, but for small T (substantially sparse regime), nearly all are unique.
# Instead, let's use the process: for small T, number of unique edges grows linearly until collisions (few for small T)
# For very sparse: L ≈ total number of edge activations per time step * T = alpha*N*m*T
# So <k>_agg = (2*L)/N = 2*alpha*m*T
# Want <k> = 0.4, thus: 0.4 = 2*alpha*m*T => T = 0.4/(2*alpha*m)
T_agg = int(round(desired_mean_degree / (2*alpha*m)))  # Should be 1

# Enforce at least T=1 for minimal static structure
T_agg = max(1, T_agg)

# Regenerate aggregated network from temporal process over T_agg timesteps
rng = np.random.default_rng(seed=2024)  # new seed for independent sampling
agg_edge_weights = {}
for t in range(T_agg):
    activations = rng.uniform(size=num_nodes) < alpha
    active_nodes = np.where(activations)[0]
    for node in active_nodes:
        partners = rng.choice([i for i in range(num_nodes) if i != node], size=m, replace=False)
        for partner in partners:
            edge = tuple(sorted((node, partner)))
            agg_edge_weights[edge] = agg_edge_weights.get(edge, 0) + 1

G_agg = nx.Graph()
G_agg.add_nodes_from(range(num_nodes))
for (i, j), w in agg_edge_weights.items():
    G_agg.add_edge(i, j, weight=w)

# Save corrected aggregated network
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
pth = os.path.join(output_dir, "activity-aggregated-network-corrected.npz")
sparse.save_npz(pth, nx.to_scipy_sparse_array(G_agg))
data_w = np.zeros((G_agg.number_of_edges(), 3), dtype=int)
for idx, (i, j, d) in enumerate(G_agg.edges(data=True)):
    data_w[idx] = [i, j, d['weight']]
np.savez_compressed(os.path.join(output_dir, "activity-aggregated-weights-corrected.npz"), edges=data_w)

# Diagnostics
deg_seq = np.array([d for n, d in G_agg.degree()])
mean_k_corr = deg_seq.mean()
mean_k2_corr = (deg_seq **2).mean()

plt.figure()
plt.hist(deg_seq, bins=np.arange(deg_seq.max()+2)-0.5, density=True, color='royalblue')
plt.xlabel('Degree')
plt.ylabel('Fraction of nodes')
plt.title('Corrected degree distribution (aggregated network)')
deg_hist_path = os.path.join(output_dir,"activity-degree-distribution-corrected.png")
plt.savefig(deg_hist_path)
plt.close()

components = [len(c) for c in nx.connected_components(G_agg)]
GCC_size = max(components) if components else 0
prop_in_GCC = GCC_size / num_nodes

# Return main diagnostics
{
    'agg_network_path': pth,
    'agg_weights_path': os.path.join(output_dir, "activity-aggregated-weights-corrected.npz"),
    'deg_hist_path': deg_hist_path,
    'mean_k_corr': mean_k_corr,
    'mean_k2_corr': mean_k2_corr,
    'gcc_size_corr': GCC_size,
    'gcc_frac_corr': prop_in_GCC,
}