
import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx
from scipy import sparse
# Reload ER network and create degree histogram
from scipy.sparse import load_npz
adj = load_npz(os.path.join(os.getcwd(), "output", "erdosrenyi-n1000-k10.npz"))
G = nx.from_scipy_sparse_array(adj)
degrees = [d for _, d in G.degree()]
plt.figure(figsize=(6,4))
plt.hist(degrees, bins=np.arange(0, max(degrees)+2)-0.5, density=True, color='skyblue', edgecolor='k')
plt.title('Erdos-Renyi Degree Distribution (N=1000, <k>=10)')
plt.xlabel('Degree')
plt.ylabel('Fraction of Nodes')
png1 = os.path.join(os.getcwd(),'output','erdosrenyi-degree-dist.png')
plt.savefig(png1)
plt.close()
return_vars=['png1']