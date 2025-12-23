
# Chain of Thought for Modeling
# 1. Analytical Understanding:
#   - We'll describe and prepare SEIR ODEs for homogeneous mixing: classic deterministic.
#   - We'll derive the next-generation matrix for heterogeneous networks (mean-field for degree classes, e.g., configuration model approach).
# 2. Network Construction:
#   - A homogeneous (Erdős-Rényi, ER) network (Poisson degree distribution: narrow)
#   - A degree-heterogeneous (Barabási–Albert, BA) scale-free network
# 3. SEIR Model Parameters:
#   - Chosen for realistic epidemic (~R0 ~ 2.2 as in COVID-19):
#     * beta (transmission rate)
#     * mu (incubation/exposed rate)
#     * gamma (recovery rate)
# 4. Simulation Size:
#   - N = 2000 nodes (computationally feasible, representative)
# 5. Both networks are generated and basic statistics printed and exported.
# 6. Degree moments (<k>, <k^2>) for later use in analytical formulas.

import networkx as nx
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import os

N = 2000  # Population size
mean_k = 8  # Typical mean degree for close-contact network

# 1. Homogeneous Mixing Network (Erdős–Rényi)
p = mean_k / (N - 1)
G_ER = nx.erdos_renyi_graph(N, p)
g_ER = nx.to_scipy_sparse_array(G_ER)

# 2. Heterogeneous Network (Barabási–Albert scale-free, m is mean degree/2)
m_BA = mean_k // 2
G_BA = nx.barabasi_albert_graph(N, m_BA)
g_BA = nx.to_scipy_sparse_array(G_BA)

# Save the networks
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
sparse.save_npz(os.path.join(output_dir, 'homogeneous_network.npz'), g_ER)
sparse.save_npz(os.path.join(output_dir, 'heterogeneous_network.npz'), g_BA)

# Compute and plot degree distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for idx, (G, title) in enumerate(zip([G_ER, G_BA], ['ER (Homogeneous)', 'BA (Heterogeneous)'])):
    degs = [d for n, d in G.degree()]
    axes[idx].hist(degs, bins=30, color='C{}'.format(idx), edgecolor='k', alpha=0.7)
    axes[idx].set_title(title)
    axes[idx].set_xlabel('Degree')
    axes[idx].set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'degree_distributions.png'))
plt.close()

# Calculate degree moments
er_degs = np.array([d for _, d in G_ER.degree()])
ba_degs = np.array([d for _, d in G_BA.degree()])
ER_k1 = er_degs.mean()
ER_k2 = (er_degs**2).mean()
BA_k1 = ba_degs.mean()
BA_k2 = (ba_degs**2).mean()

# Save degree moments
with open(os.path.join(output_dir, 'degree_moments.txt'), 'w') as f:
    print(f"ER: <k>={ER_k1:.2f}, <k^2>={ER_k2:.2f}", file=f)
    print(f"BA: <k>={BA_k1:.2f}, <k^2>={BA_k2:.2f}", file=f)

# Return paths and statistics for reporting and downstream steps
network_paths = [os.path.join('output', 'homogeneous_network.npz'), os.path.join('output', 'heterogeneous_network.npz')]
plot_paths = {'output/degree_distributions.png': 'Degree distributions for ER (homogeneous) and BA (heterogeneous) networks.'}
network_details = {'ER': {'mean_degree': ER_k1, 'degree2_moment': ER_k2},
                  'BA': {'mean_degree': BA_k1, 'degree2_moment': BA_k2}}
(network_paths, plot_paths, network_details)
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse

# --- PLAN ---
# 1. Community network: Let's use 4 communities of size 250 each (total N=1000), high intra-community, sparse inter-community
sizes = [250, 250, 250, 250]
p_intra = 0.08   # Dense within community
p_inter = 0.005  # Sparse between communities

# 2. Create stochastic block model (SBM)
probs = np.full((4, 4), p_inter)
np.fill_diagonal(probs, p_intra)
G = nx.stochastic_block_model(sizes, probs, seed=42)

# 3. Save network as sparse matrix
os.makedirs(os.path.join(os.getcwd(), "output"), exist_ok=True)
sparse.save_npz(os.path.join(os.getcwd(), "output", "network.npz"), nx.to_scipy_sparse_array(G))

# 4. Plot degree distribution
plt.figure()
deg = [d for n, d in G.degree()]
plt.hist(deg, bins=30, color='skyblue', edgecolor='k')
plt.xlabel("Degree")
plt.ylabel("Number of nodes")
plt.title("Degree Distribution: SBM Community Network")
plt.savefig(os.path.join(os.getcwd(), "output", "degree_dist_sbm.png"), dpi=150)
plt.close()

# 5. Calculate <k> and <k^2>
deg_array = np.array(deg)
mean_k = deg_array.mean()
mean_k2 = (deg_array**2).mean()
mean_k, mean_k2