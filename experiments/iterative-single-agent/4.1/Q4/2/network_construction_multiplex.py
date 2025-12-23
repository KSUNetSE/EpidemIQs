
import networkx as nx
import scipy.sparse as sparse
import os
from networkx.generators.random_graphs import erdos_renyi_graph
import matplotlib.pyplot as plt
# Multiplex layers
N = 100
# Layer A: ER p=0.05, Layer B: ER p=0.03
G_A = erdos_renyi_graph(N, 0.05, seed=1)
G_B = erdos_renyi_graph(N, 0.03, seed=2)
A = nx.to_scipy_sparse_array(G_A)
B = nx.to_scipy_sparse_array(G_B)
sparse.save_npz(os.path.join(os.getcwd(), "output", "network_layerA.npz"), A)
sparse.save_npz(os.path.join(os.getcwd(), "output", "network_layerB.npz"), B)
# Degree distribution, plot
degrees_A = [d for n, d in G_A.degree()]
degrees_B = [d for n, d in G_B.degree()]
plt.figure()
plt.hist(degrees_A, alpha=0.5, label='Layer A')
plt.hist(degrees_B, alpha=0.5, label='Layer B')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Degree Distribution of Multiplex Layers')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), "output", "degree_dist_multiplex.png"))
plt.close()
# Mean degree <k> and <k^2>
mean_deg_A = sum(degrees_A)/N
mean_deg_B = sum(degrees_B)/N
second_deg_A = sum([k**2 for k in degrees_A])/N
second_deg_B = sum([k**2 for k in degrees_B])/N
network_details = {'mean_deg_A': mean_deg_A, 'mean_deg_B': mean_deg_B, 'second_deg_A': second_deg_A, 'second_deg_B': second_deg_B}
