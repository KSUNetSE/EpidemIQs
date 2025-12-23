
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import os
np.random.seed(123)

# Parameters
N = 200  # number of nodes
# Layer A: Scale-free network for Virus 1
layer_A = nx.barabasi_albert_graph(N, 5)
# Layer B: Random (Erdos-Renyi) with low correlation
layer_B = nx.erdos_renyi_graph(N, 0.07)

# (Optional: Visualize layers' overlap in high-degree nodes for report)
deg_A = np.array([layer_A.degree(n) for n in range(N)])
deg_B = np.array([layer_B.degree(n) for n in range(N)])

plt.figure(figsize=(6,5))
plt.scatter(deg_A, deg_B, alpha=0.5)
plt.xlabel('Degree (Layer A, BA)')
plt.ylabel('Degree (Layer B, ER)')
plt.title('Scatter Degree: Layer A vs Layer B')
plt.grid(True)
plt.tight_layout()
# Save figure
os.makedirs(os.path.join(os.getcwd(), 'output'), exist_ok=True)
plt.savefig(os.path.join(os.getcwd(), 'output', 'deg_scatter-01.png'))

# Save layers as sparse matrices
A_csr = nx.to_scipy_sparse_array(layer_A)
B_csr = nx.to_scipy_sparse_array(layer_B)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'layer_A-01.npz'), A_csr)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'layer_B-01.npz'), B_csr)
# Network diagnostics
mean_deg_A = np.mean(deg_A)
k2_A = np.mean(deg_A**2)
mean_deg_B = np.mean(deg_B)
k2_B = np.mean(deg_B**2)

(mean_deg_A, k2_A, mean_deg_B, k2_B)