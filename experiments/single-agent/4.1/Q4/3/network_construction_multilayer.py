
# Plan:
# 1. Construct two independent network layers A and B (multiplex: same nodes, different edges).
# 2. Ensure little-to-no overlap in centrality, per analytic results (e.g. node degrees in A uncorrelated with node degrees in B).
# 3. Save the two layers as separate CSR npz files.
# 4. Visualize their degree distributions, and correlation between degrees across layers, for multiplex justification.
# 5. Compute mean degree and second moment for each layer.

import networkx as nx
import numpy as np
import os
from scipy import sparse
import matplotlib.pyplot as plt

N = 1000  # number of nodes
mean_deg_A = 8
mean_deg_B = 8
# Layer A: Barabasi-Albert
G_A = nx.barabasi_albert_graph(N, mean_deg_A//2)
# Layer B: Configuration model, shuffled degrees of A to avoid correlation
degrees = np.array([d for n,d in G_A.degree()])
np.random.shuffle(degrees)
# Ensure even sum for configuration model
if sum(degrees)%2 != 0:
    degrees[0] += 1
G_B = nx.configuration_model(degrees.tolist(), seed=42)
G_B = nx.Graph(G_B) # remove parallel edges/self-loops
G_B.remove_edges_from(nx.selfloop_edges(G_B))

# Save networks
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
sparse.save_npz(os.path.join(output_dir, 'networkA.npz'), nx.to_scipy_sparse_array(G_A))
sparse.save_npz(os.path.join(output_dir, 'networkB.npz'), nx.to_scipy_sparse_array(G_B))

# Degree distributions
plt.figure(figsize=(8,4))
plt.hist([d for n,d in G_A.degree()], bins=30, alpha=0.6, label='Layer A (BA)')
plt.hist([d for n,d in G_B.degree()], bins=30, alpha=0.6, label='Layer B (Config.)')
plt.legend()
plt.title('Degree Distributions of Network Layers A and B')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.tight_layout()
deg_hist_path = os.path.join(output_dir, 'deg_dist_layers.png')
plt.savefig(deg_hist_path)
plt.close()

# Degree correlation between layers
degA = np.array([G_A.degree(n) for n in range(N)])
degB = np.array([G_B.degree(n) for n in range(N)])
degree_corr = np.corrcoef(degA, degB)[0,1]
plt.figure(figsize=(5,5))
plt.scatter(degA, degB, s=5, alpha=0.5)
plt.xlabel('Degree in Layer A')
plt.ylabel('Degree in Layer B')
plt.title(f'Degree-Degree Correlation\nPearson r={degree_corr:.3f}')
deg_corr_path = os.path.join(output_dir, 'deg_corr.png')
plt.savefig(deg_corr_path)
plt.close()

mean_deg_A = degA.mean()
mean_deg2_A = (degA**2).mean()
mean_deg_B = degB.mean()
mean_deg2_B = (degB**2).mean()

results = {'networkA_path': os.path.join(output_dir, 'networkA.npz'),
           'networkB_path': os.path.join(output_dir, 'networkB.npz'),
           'deg_hist_path': deg_hist_path,
           'deg_corr_path': deg_corr_path,
           'mean_deg_A': mean_deg_A,
           'mean_deg2_A': mean_deg2_A,
           'mean_deg_B': mean_deg_B,
           'mean_deg2_B': mean_deg2_B,
           'degree_corr': degree_corr}
results