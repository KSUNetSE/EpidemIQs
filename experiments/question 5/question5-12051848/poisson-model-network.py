
import numpy as np
import networkx as nx
from scipy import sparse
import matplotlib.pyplot as plt
import os

# Step 1: Parameters
N = 10000  # number of nodes
z = 3      # mean degree
q = 4      # mean excess degree
k_targets = [10]  # for targeted strategy demo
random_seed = 42  # reproducibility
np.random.seed(random_seed)

# Step 2: Determine degree distribution
# Poisson distributed degree with lambda = 3 yields q = 4
lmbda = z
poisson_degrees = np.random.poisson(lam=lmbda, size=N)
# Ensure sum(degrees) is even for configuration model
if poisson_degrees.sum() % 2 == 1:
    idx = np.random.randint(0, N)
    poisson_degrees[idx] += 1

# Step 3: Build the configuration model network
G_pois = nx.configuration_model(poisson_degrees, seed=random_seed)
# Remove parallel edges and self-loops to get a simple graph
G_pois = nx.Graph(G_pois)
G_pois.remove_edges_from(nx.selfloop_edges(G_pois))

# Step 4: Measure degree distribution and centralities
actual_degrees = [d for n, d in G_pois.degree()]
mean_k = np.mean(actual_degrees)
mean_k2 = np.mean(np.array(actual_degrees)**2)
q_emp = (mean_k2 - mean_k) / mean_k
# Number and fraction of degree-10 nodes (for the targeted strategy demo)
num_deg10 = sum([1 for k in actual_degrees if k == 10])
fraction_deg10 = num_deg10 / N
# Largest connected component (LCC) size
lcc = max(nx.connected_components(G_pois), key=len)
size_lcc = len(lcc)
# Degree assortativity (should be near zero for uncorrelated)
assortativity = nx.degree_assortativity_coefficient(G_pois)
# Clustering coefficient (should be low)
avg_clustering = nx.average_clustering(G_pois)

# Step 5: Save the network
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, "poisson3-z3-q4-network.npz")
sparse.save_npz(network_path, nx.to_scipy_sparse_array(G_pois))

# Step 6: Plot degree distribution
plt.figure(figsize=(8,6))
degree_counts, bins, _ = plt.hist(actual_degrees, bins=range(0, max(actual_degrees)+2),
                                 density=True, color="skyblue", edgecolor="k", alpha=0.7)
plt.title("Degree Distribution (Poisson, z=3, N=10000)")
plt.xlabel("Degree k")
plt.ylabel("P(k)")
dist_plot_path = os.path.join(output_dir, "degree-distribution-poisson3-z3-q4.png")
plt.savefig(dist_plot_path)
plt.close()

# Step 7: Diagnostics (output summary)
metrics = {
    'mean_degree': mean_k,
    'second_moment': mean_k2,
    'mean_excess_degree': q_emp,
    'num_deg10': num_deg10,
    'fraction_deg10': fraction_deg10,
    'size_lcc': size_lcc,
    'assortativity': assortativity,
    'clustering': avg_clustering,
    'network_file': network_path,
    'degree_dist_plot': dist_plot_path
}

metrics