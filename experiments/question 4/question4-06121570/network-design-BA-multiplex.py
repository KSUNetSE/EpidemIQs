
import os
import networkx as nx
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Step: Generate multiplex Scale-free (BA) networks with maximal and minimal interlayer degree correlation
N = 1000
m = 4 # Gives approximately <k> ~ 2m

os.makedirs("output", exist_ok=True)

# Layer A: BA network
layerA = nx.barabasi_albert_graph(N, m, seed=99)

# Layer B (max correlated): Same as layerA
layerB_corr = layerA.copy()

# Layer B (minimally correlated): BA network with degree anti-correlated to A
# Rank nodes by degree in LayerA
A_deg = np.array([layerA.degree(n) for n in range(N)])
sortA = np.argsort(A_deg)[::-1] # Descending degree order

# Generate a new independent BA graph for layer B
layerB_rand = nx.barabasi_albert_graph(N, m, seed=101)
B_deg = np.array([layerB_rand.degree(n) for n in range(N)])
sortB = np.argsort(B_deg)      # Ascending degree order

# Permute layerB_rand node labels to anti-correlate with layerA degrees
mapping = {int(sortB[i]): int(sortA[i]) for i in range(N)}
layerB_anticorr = nx.relabel_nodes(layerB_rand, mapping)

# Degree/correlation stats
A_degrees = np.array([deg for n, deg in layerA.degree()])
B_corr_degrees = np.array([deg for n, deg in layerB_corr.degree()])
B_anticorr_degrees = np.array([deg for n, deg in sorted(layerB_anticorr.degree(), key=lambda x: x[0])])

A_mean = np.mean(A_degrees)
A_k2 = np.mean(A_degrees ** 2)
B_corr_mean = np.mean(B_corr_degrees)
B_corr_k2 = np.mean(B_corr_degrees ** 2)
B_anticorr_mean = np.mean(B_anticorr_degrees)
B_anticorr_k2 = np.mean(B_anticorr_degrees ** 2)

corr_same = spearmanr(A_degrees, B_corr_degrees).correlation
corr_anticorr = spearmanr(A_degrees, B_anticorr_degrees).correlation

# Save adjacency matrices
sparse.save_npz(os.path.join("output", "layerA-BAMux.npz"), nx.to_scipy_sparse_array(layerA))
sparse.save_npz(os.path.join("output", "layerB-BAMux-maxcorr.npz"), nx.to_scipy_sparse_array(layerB_corr))
sparse.save_npz(os.path.join("output", "layerB-BAMux-anticorr.npz"), nx.to_scipy_sparse_array(layerB_anticorr))

# Plots
bins = np.unique(A_degrees)
bins = np.append(bins, bins[-1]+1)
plt.figure(figsize=(8,5))
plt.hist(A_degrees, bins=bins, alpha=0.7, label="Layer A (BA)")
plt.hist(B_anticorr_degrees, bins=bins, alpha=0.7, label="Layer B (BA, anticorr)")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.legend()
plt.title("Degree Distribution BA Multiplex (LayerA & Anticorrelated LayerB)")
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig("output/BA-degree-multiplex.png")
plt.close()

plt.scatter(A_degrees, B_anticorr_degrees, alpha=0.5)
plt.xlabel("LayerA Degree")
plt.ylabel("LayerB Degree (Anti-corr)")
plt.title("Interlayer Degree Scatter: BA Anticorrelated Layers\nSpearman rho = {:.2f}".format(corr_anticorr))
plt.tight_layout()
plt.savefig("output/BA-degree-scatter-anticorr.png")
plt.close()

plt.scatter(A_degrees, B_corr_degrees, alpha=0.5)
plt.xlabel("LayerA Degree")
plt.ylabel("LayerB Degree (Max Corr)")
plt.title("Interlayer Degree Scatter: BA Maximally Correlated\nSpearman rho = {:.2f}".format(corr_same))
plt.tight_layout()
plt.savefig("output/BA-degree-scatter-maxcorr.png")
plt.close()
