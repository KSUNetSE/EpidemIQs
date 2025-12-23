
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import sparse
np.random.seed(42)

# Parameters
N = 1000
# Complete (homogeneous) network
g_complete = nx.complete_graph(N)
fn = os.path.join(os.getcwd(), "output", "completegraphnetwork.npz")
sparse.save_npz(fn, nx.to_scipy_sparse_array(g_complete))

# Degree stats
degrees_complete = np.array([d for n, d in g_complete.degree()])
meank_complete = degrees_complete.mean()
meank2_complete = (degrees_complete**2).mean()

# Plot degree histogram
plt.figure(figsize=(6,4))
plt.hist(degrees_complete, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
plt.xlabel("Degree")
plt.ylabel("Count")
plt.title("Degree Distribution: Complete Graph")
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), "output", "degreehistogram-completegraph.png"))
plt.close()

# Return paths and stats
complete_paths = [fn]
complete_plots = {"output/degreehistogram-completegraph.png": "Degree distribution of homogeneous-mixing (complete) network"}
complete_degree_stats = (meank_complete, meank2_complete)
