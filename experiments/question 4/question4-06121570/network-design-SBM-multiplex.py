
import os
import networkx as nx
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# SBM multiplex: examining strong vs. weak community overlap
N = 1000
num_communities = 6
sizes = [N // num_communities] * num_communities
for i in range(N % num_communities):
    sizes[i] += 1
within = 0.10
between = 0.01
p_matrix = np.full((num_communities, num_communities), between)
np.fill_diagonal(p_matrix, within)

os.makedirs("output", exist_ok=True)

# SBM LayerA: Assign base communities
layerA = nx.stochastic_block_model(sizes, p_matrix, seed=202)
comm_labels_A = []
for i, s in enumerate(sizes):
    comm_labels_A += [i]*s
comm_labels_A = np.array(comm_labels_A)

# LayerB, strong overlap: same community assignment
layerB_highoverlap = nx.stochastic_block_model(sizes, p_matrix, seed=303)
comm_labels_B_high = comm_labels_A.copy()

# LayerB, partial overlap: shuffle a fraction of nodes to different communities
comm_labels_B_partial = comm_labels_A.copy()
permute_fraction = 0.4
num_permute = int(N * permute_fraction)
np.random.seed(404)
shuffled = np.random.choice(N, size=num_permute, replace=False)
for idx in shuffled:
    comm_labels_B_partial[idx] = np.random.choice(np.delete(np.arange(num_communities), comm_labels_A[idx]))
layerB_partialoverlap = nx.stochastic_block_model(
    [np.sum(comm_labels_B_partial==i) for i in range(num_communities)],
    p_matrix, seed=405)

# Degree stats (use after relabel if needed)
A_degrees = np.array([deg for n, deg in layerA.degree()])
B_high_degrees = np.array([deg for n, deg in layerB_highoverlap.degree()])
B_partial_degrees = np.array([deg for n, deg in layerB_partialoverlap.degree()])

A_mean = np.mean(A_degrees)
A_k2 = np.mean(A_degrees**2)
B_high_mean = np.mean(B_high_degrees)
B_high_k2 = np.mean(B_high_degrees**2)
B_partial_mean = np.mean(B_partial_degrees)
B_partial_k2 = np.mean(B_partial_degrees**2)

# Community overlap score: fraction of nodes w/ same label in both layers
same_count = np.sum(comm_labels_A == comm_labels_B_partial)
overlap_frac = same_count / N

# Save networks
sparse.save_npz(os.path.join("output", "layerA-SBMMux.npz"), nx.to_scipy_sparse_array(layerA))
sparse.save_npz(os.path.join("output", "layerB-SBMMux-highoverlap.npz"), nx.to_scipy_sparse_array(layerB_highoverlap))
sparse.save_npz(os.path.join("output", "layerB-SBMMux-partialoverlap.npz"), nx.to_scipy_sparse_array(layerB_partialoverlap))

bins = np.arange(0, max(A_degrees.max(), B_high_degrees.max(), B_partial_degrees.max())+2) - 0.5
plt.hist(A_degrees, bins=bins, alpha=0.6, label="LayerA (SBM)")
plt.hist(B_partial_degrees, bins=bins, alpha=0.6, label="LayerB Partial Overlap")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.legend()
plt.title("Degree Distribution for SBM Multiplex (Partial Overlap)")
plt.savefig("output/SBM-degree-partialoverlap.png")
plt.close()

plt.hist(A_degrees, bins=bins, alpha=0.6, label="LayerA (SBM)")
plt.hist(B_high_degrees, bins=bins, alpha=0.6, label="LayerB High Overlap")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.legend()
plt.title("Degree Distribution for SBM Multiplex (High Overlap)")
plt.savefig("output/SBM-degree-highoverlap.png")
plt.close()
