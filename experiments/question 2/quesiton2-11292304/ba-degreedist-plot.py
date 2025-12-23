
import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx
from scipy import sparse
from scipy.sparse import load_npz
adj = load_npz(os.path.join(os.getcwd(), "output", "barabasialbert-n1000-k10.npz"))
bag = nx.from_scipy_sparse_array(adj)
degrees_ba = [d for _, d in bag.degree()]
plt.figure(figsize=(6,4))
plt.hist(degrees_ba, bins=np.arange(0, max(degrees_ba)+2)-0.5, density=True, color='tomato', edgecolor='k', log=True)
plt.title('Barabasi-Albert Degree Distribution (N=1000, <k>=10)')
plt.xlabel('Degree')
plt.ylabel('Fraction of Nodes')
png2 = os.path.join(os.getcwd(),'output','barabasialbert-degree-dist.png')
plt.savefig(png2)
plt.close()
return_vars=['png2']