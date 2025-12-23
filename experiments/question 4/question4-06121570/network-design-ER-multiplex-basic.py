
import os
import numpy as np
import networkx as nx
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Setup
N = 1000
mean_degree = 8
edge_prob = mean_degree / (N - 1)

os.makedirs("output", exist_ok=True)

# ER Multiplex: maximally correlated vs. uncorrelated
# Layer A (ER)
layerA = nx.erdos_renyi_graph(N, edge_prob, seed=42)
# Layer B (ER): Maximally correlated (copy)
layerB_corr = layerA.copy()
# Layer B (ER): Uncorrelated (independent)
layerB_uncorr = nx.erdos_renyi_graph(N, edge_prob, seed=24)

A_degrees = np.array([deg for n, deg in layerA.degree()])
B_corr_degrees = np.array([deg for n, deg in layerB_corr.degree()])
B_uncorr_degrees = np.array([deg for n, deg in layerB_uncorr.degree()])

A_mean = np.mean(A_degrees)
A_k2 = np.mean(A_degrees ** 2)
B_corr_mean = np.mean(B_corr_degrees)
B_corr_k2 = np.mean(B_corr_degrees ** 2)
B_uncorr_mean = np.mean(B_uncorr_degrees)
B_uncorr_k2 = np.mean(B_uncorr_degrees ** 2)

corr_same = spearmanr(A_degrees, B_corr_degrees).correlation
corr_uncorr = spearmanr(A_degrees, B_uncorr_degrees).correlation

sparse.save_npz(os.path.join("output", "layerA-ERMux.npz"), nx.to_scipy_sparse_array(layerA))
sparse.save_npz(os.path.join("output", "layerB-ERMux-maxcorr.npz"), nx.to_scipy_sparse_array(layerB_corr))
sparse.save_npz(os.path.join("output", "layerB-ERMux-uncorr.npz"), nx.to_scipy_sparse_array(layerB_uncorr))

plt.hist(A_degrees, bins=25, alpha=0.7, label="Layer A")
plt.hist(B_uncorr_degrees, bins=25, alpha=0.7, label="Layer B Uncorr.")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.legend()
plt.title("Degree Distributions in ER Multiplex (Layer A & B Uncorrelated)")
plt.savefig("output/ER-degree-multiplex.png")
plt.close()

plt.scatter(A_degrees, B_uncorr_degrees, alpha=0.6)
plt.xlabel("LayerA Degree")
plt.ylabel("LayerB Degree (Uncorrelated)")
plt.title("Interlayer Degree Scatter: ER Uncorrelated\nSpearman rho = {:.2f}".format(corr_uncorr))
plt.savefig("output/ER-degree-scatter-uncorr.png")
plt.close()

plt.scatter(A_degrees, B_corr_degrees, alpha=0.6)
plt.xlabel("LayerA Degree")
plt.ylabel("LayerB Degree (Max Corr)")
plt.title("Interlayer Degree Scatter: ER Maximally Correlated\nSpearman rho = {:.2f}".format(corr_same))
plt.savefig("output/ER-degree-scatter-maxcorr.png")
plt.close()
