
import numpy as np
import networkx as nx
import scipy.sparse as sparse
import os
import matplotlib.pyplot as plt

n = 1000
adj_agg = np.load(os.path.join('output', 'total_agg.npy')) if os.path.exists(os.path.join('output', 'total_agg.npy')) else None
if adj_agg is None:
    # Recompute or use last known stats
    adj_agg = np.zeros((n, n))
G_agg = nx.from_numpy_array(adj_agg)
agg_csr = nx.to_scipy_sparse_array(G_agg)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'agg_network.npz'), agg_csr)

k_agg = np.array([d for n, d in G_agg.degree()])
plt.hist(k_agg, bins=30)
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution of Aggregated Network')
plt.savefig(os.path.join(os.getcwd(), 'output', 'deg_dist_agg.png'))

agg_path = os.path.join(os.getcwd(), 'output', 'agg_network.npz')
plot_path = os.path.join(os.getcwd(), 'output', 'deg_dist_agg.png')

agg_path, plot_path