
import os, numpy as np, networkx as nx, scipy.sparse as sparse, json
# Load networks saved earlier
G_er_csr = sparse.load_npz(os.path.join(os.getcwd(),'output','network_er.npz'))
G_ba_csr = sparse.load_npz(os.path.join(os.getcwd(),'output','network_ba.npz'))
# Convert to degrees
k_er = np.array(G_er_csr.sum(axis=1)).flatten()
k_ba = np.array(G_ba_csr.sum(axis=1)).flatten()
mean_er = k_er.mean(); mean2_er = (k_er**2).mean()
mean_ba = k_ba.mean(); mean2_ba = (k_ba**2).mean()
