
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import os

# Set parameters for the population and network
N = 500  # Network size
p = 0.05 # Edge probability for ER network, to get reasonable <k>
np.random.seed(42)  # reproducibility

# Create ER random graph
G = nx.erdos_renyi_graph(N, p)

# Save the network in sparse format
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
sparse.save_npz(network_path, nx.to_scipy_sparse_array(G))

# Compute and display <k> and <k^2>
degrees = np.array([deg for n, deg in G.degree()])
mean_k = degrees.mean()
mean_k2 = np.mean(degrees ** 2)

# Plot degree distribution
plt.figure(figsize=(7, 4))
plt.hist(degrees, bins=range(degrees.min(), degrees.max() + 2), color='steelblue', alpha=0.8)
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.title('Degree Distribution (Erdős-Rényi, N=500, p=0.05)')
plt.grid(True)
plot_path = os.path.join(os.getcwd(), 'output', 'network_degree_dist.png')
plt.savefig(plot_path)
plt.close()

# Return key results for modeling phase
(mean_k, mean_k2, network_path, plot_path)
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import os

current_directory = os.getcwd()
output_dir = os.path.join(current_directory, "output")
os.makedirs(output_dir, exist_ok=True)

# 1. SIR Configuration (chain breaking due to decline in infectives)
N = 1000
p = 0.01  # ER probability to give moderate connectivity
G_sir = nx.erdos_renyi_graph(N, p, seed=123)
sparse.save_npz(os.path.join(output_dir, "network_sir.npz"), nx.to_scipy_sparse_array(G_sir))

# Plotting degree distribution SIR
degrees = [d for n, d in G_sir.degree()]
plt.figure(figsize=(5,4))
plt.hist(degrees, bins=30)
plt.title("Degree Distribution SIR (ER Network)")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "degree_dist_sir.png"))
plt.close()

mean_k = np.mean(degrees)
mean_k2 = np.mean(np.square(degrees))

# 2. SIS Configuration (compare to see non-extinction if R0>1)
G_sis = nx.erdos_renyi_graph(N, p, seed=456)
sparse.save_npz(os.path.join(output_dir, "network_sis.npz"), nx.to_scipy_sparse_array(G_sis))

# Plotting degree distribution SIS
degrees_sis = [d for n, d in G_sis.degree()]
plt.figure(figsize=(5,4))
plt.hist(degrees_sis, bins=30)
plt.title("Degree Distribution SIS (ER Network)")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "degree_dist_sis.png"))
plt.close()

mean_k_sis = np.mean(degrees_sis)
mean_k2_sis = np.mean(np.square(degrees_sis))

# Return network metrics
metrics = {'mean_k': mean_k, 'mean_k2': mean_k2, 'mean_k_sis': mean_k_sis, 'mean_k2_sis': mean_k2_sis,
           'sir_path': os.path.join(output_dir, "network_sir.npz"),
           'sis_path': os.path.join(output_dir, "network_sis.npz"),
           'sir_deg_plot': os.path.join(output_dir, "degree_dist_sir.png"),
           'sis_deg_plot': os.path.join(output_dir, "degree_dist_sis.png")}
metrics