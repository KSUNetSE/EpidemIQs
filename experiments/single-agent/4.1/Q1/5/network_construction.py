
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import os

# Set parameters
N = 1000  # population size
ER_p = 0.01  # ER probability for homogeneous mixing network
BA_m = 3     # Barabasi-Albert parameter for heterogeneous network

# 1. Homogeneous (Erdos-Renyi) network
G_hom = nx.erdos_renyi_graph(N, ER_p)
mean_deg_hom = np.mean([d for _, d in G_hom.degree()])
k2_hom = np.mean([d ** 2 for _, d in G_hom.degree()])
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network_hom.npz'), nx.to_scipy_sparse_array(G_hom))

# 2. Heterogeneous (Barabasi-Albert) network
G_het = nx.barabasi_albert_graph(N, BA_m)
mean_deg_het = np.mean([d for _, d in G_het.degree()])
k2_het = np.mean([d ** 2 for _, d in G_het.degree()])
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network_het.npz'), nx.to_scipy_sparse_array(G_het))

# Make Degree Distribution Plots
plt.figure(figsize=(10,5))
deghist_hom = [d for n, d in G_hom.degree()]
deghist_het = [d for n, d in G_het.degree()]
plt.hist(deghist_hom, bins=30, alpha=0.5, label='ER Network (Homogeneous)')
plt.hist(deghist_het, bins=30, alpha=0.5, label='BA Network (Heterogeneous)')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.title('Degree Distributions (Homogeneous vs. Heterogeneous Network)')
plt.legend()
plt.savefig(os.path.join(os.getcwd(), 'output', 'degree_dist.png'))
plt.close()

# Return stats for further use
(mean_deg_hom, k2_hom, mean_deg_het, k2_het)
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import sparse

# Parameters for network construction based on standard SIR literature
# Let's make a static network with N=1000 nodes and mean degree <k>=8 (moderately connected, useful for SIR spread test)
N = 1000
mean_k = 8
p = mean_k / (N-1)

# Generate Erdős-Rényi random graph (ER)
G = nx.erdos_renyi_graph(N, p)

# Save the network
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, "network.npz")
sparse.save_npz(network_path, nx.to_scipy_sparse_array(G))

# Degree metrics
degrees = np.array([d for n, d in G.degree()])
mean_degree = degrees.mean()
second_moment = (degrees**2).mean()

# Degree histogram for appendices
plt.figure()
plt.hist(degrees, bins=30, edgecolor='black')
plt.title('Degree Distribution of ER Network')
plt.xlabel('Degree')
plt.ylabel('Count')
degree_hist_path = os.path.join(output_dir, 'degree_dist.png')
plt.savefig(degree_hist_path)
plt.close()

# Return info for following phase
(network_path, mean_degree, second_moment, degree_hist_path)
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import sparse

# Set parameters based on common SIR over static networks
N = 1000  # population size
mean_degree = 10  # typical for human-sized social groups
p = mean_degree / (N - 1)  # ER connection prob

# Generate the network
G = nx.erdos_renyi_graph(N, p)
# Save as sparse matrix
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
sparse.save_npz(os.path.join(output_dir, "network.npz"), nx.to_scipy_sparse_array(G))
# Compute degree statistics
k_vals = [d for n, d in G.degree()]
mean_k = np.mean(k_vals)
k2 = np.mean(np.array(k_vals)**2)
# Plot degree distribution
plt.figure(figsize=(6,4))
plt.hist(k_vals, bins=np.arange(0,max(k_vals)+1)-0.5, rwidth=0.85)
plt.xlabel("Degree")
plt.ylabel("# Nodes")
plt.title("Degree Distribution (Erdos-Renyi, N=1000, <k>=10)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'degree_dist.png'))
plt.close()

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import sparse
# 1. Set parameters for scale-free network (Barabási-Albert) simulation
N = 1000  # Size of network
m = 4     # Each new node attaches to m existing nodes
seed = 42
G = nx.barabasi_albert_graph(N, m, seed=seed)
# 2. Plot degree distribution
degrees = [deg for n, deg in G.degree()]
plt.figure()
plt.hist(degrees, bins=range(min(degrees), max(degrees)+1), density=True, alpha=0.75)
plt.title('Degree Distribution of Barabási-Albert Network')
plt.xlabel('Degree')
plt.ylabel('Fraction of Nodes')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'degree-dist.png'))
plt.close()
# 3. Save network
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network.npz'), nx.to_scipy_sparse_array(G))
# 4. Compute mean degree <k> and second moment <k^2>
degr = np.array(degrees)
mean_k = np.mean(degr)
mean_k2 = np.mean(degr**2)
(mean_k, mean_k2)
# Modeling Phase: Constructing a static Erdős-Rényi network and calculating required statistics
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import os

# --- PARAMETERS (from literature and search) ---
N = 300  # Number of nodes
mean_degree = 6  # Typical average degree in literature for SIR studies
p = mean_degree / (N-1)  # Probability of edge creation in ER network

# --- Generate Erdős-Rényi network ---
G = nx.erdos_renyi_graph(N, p, seed=42)

# Save adjacency matrix as scipy sparse array
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
graph_path = os.path.join(output_dir, "network.npz")
sparse.save_npz(graph_path, nx.to_scipy_sparse_array(G))

# Plot degree distribution, save to file
degrees = np.array([d for n, d in G.degree()])
plt.figure(figsize=(8,4))
plt.hist(degrees, bins=range(np.min(degrees), np.max(degrees)+2), color='skyblue', density=True, rwidth=0.8)
plt.xlabel('Degree')
plt.ylabel('Fraction of Nodes')
plt.title('Degree Distribution of ER Network (N=300)')
dist_path = os.path.join(output_dir, "degree_dist.png")
plt.savefig(dist_path, dpi=200)
plt.close()

# Calculate <k> and <k^2>
mean_k = degrees.mean()
second_moment_k = (degrees ** 2).mean()

# For downstream use:
network_metrics = {'mean_degree': float(mean_k), 'second_degree_moment': float(second_moment_k)}
