
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy import sparse

# Parameters based on context and literature
num_students = 763
num_dorms = 8
sizes = [(num_students // num_dorms)] * num_dorms
for i in range(num_students % num_dorms):
    sizes[i] += 1  # Distribute remainder as evenly as possible
block_labels = []
for idx, sz in enumerate(sizes):
    block_labels += [idx] * sz
block_labels = np.array(block_labels)
# Intra- and inter-dormitory connection probabilities
p_intra = 0.25  # dense within dorms
p_inter = 0.035 # moderate cross-dorm contact
# Construct block probability matrix
prob_matrix = np.full((num_dorms, num_dorms), p_inter)
np.fill_diagonal(prob_matrix, p_intra)
# Generate stochastic block model
np.random.seed(42)
G = nx.stochastic_block_model(sizes, prob_matrix, seed=42)

# Save network for simulation
network_path = os.path.join(os.getcwd(), "output", "boarding-school-sbm-network.npz")
if not os.path.exists(os.path.dirname(network_path)):
    os.makedirs(os.path.dirname(network_path))
sparse.save_npz(network_path, nx.to_scipy_sparse_array(G))

# Diagnostics
N = G.number_of_nodes()
K = np.array([deg for n, deg in G.degree()])
mean_k = np.mean(K)
k2 = np.mean(K**2)
clustering = nx.average_clustering(G)
if nx.is_connected(G):
    lcc = N
else:
    lcc = len(max(nx.connected_components(G), key=len))
assortativity = nx.degree_assortativity_coefficient(G)

# Plot degree distribution
plt.figure(figsize=(7,5))
plt.hist(K, bins=range(K.min(), K.max()+2), color="steelblue", alpha=0.8, density=True)
plt.xlabel("Degree")
plt.ylabel("Fraction of nodes")
plt.title("Boarding School SBM Degree Distribution")
plt.tight_layout()
deg_plot_path = os.path.join(os.getcwd(), "output", "boarding-school-degree-distribution.png")
plt.savefig(deg_plot_path)
plt.close()

# Plot clustering coefficient distribution
clustvals = list(nx.clustering(G).values())
plt.figure(figsize=(7,5))
plt.hist(clustvals, bins=30, color="orange", alpha=0.7, density=True)
plt.xlabel("Clustering coefficient")
plt.ylabel("Fraction of nodes")
plt.title("Clustering Coefficient Distribution")
plt.tight_layout()
clust_plot_path = os.path.join(os.getcwd(), "output", "boarding-school-clustering-distribution.png")
plt.savefig(clust_plot_path)
plt.close()

# Save code for documentation
code_save_path = os.path.join(os.getcwd(), "output", "boarding-school-sbm-network-design.py")
with open(code_save_path, "w") as f:
    f.write('''import numpy as np\nimport networkx as nx\nfrom scipy import sparse\nnum_students = 763\nnum_dorms = 8\nsizes = [(num_students // num_dorms)] * num_dorms\nfor i in range(num_students % num_dorms):\n    sizes[i] += 1\np_intra = 0.25\np_inter = 0.035\nprob_matrix = np.full((num_dorms, num_dorms), p_inter)\nnp.fill_diagonal(prob_matrix, p_intra)\nG = nx.stochastic_block_model(sizes, prob_matrix, seed=42)\nsparse.save_npz("boarding-school-sbm-network.npz", nx.to_scipy_sparse_array(G))\n''')

# Return all relevant quantities
out_dict = {
    'network_path': network_path,
    'N': N,
    'num_edges': G.number_of_edges(),
    'num_dorms': num_dorms,
    'sizes': sizes,
    'p_intra': p_intra,
    'p_inter': p_inter,
    'mean_k': mean_k,
    'k2': k2,
    'clustering': clustering,
    'assortativity': assortativity,
    'lcc': lcc,
    'deg_plot_path': deg_plot_path,
    'clust_plot_path': clust_plot_path,
    'code_save_path': code_save_path,
}
out_dict