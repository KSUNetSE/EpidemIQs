
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

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import os

# Parameters for the network
N = 1000  # population size
k_avg = 8  # Typical average degree for epidemic scenarios
p = k_avg/(N-1)  # ER probability
np.random.seed(42)

# Create the Erdős-Rényi (ER) random graph
G = nx.erdos_renyi_graph(N, p, seed=42)

# Save to sparse matrix
if not os.path.exists('output'):
    os.makedirs('output')
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network.npz'), nx.to_scipy_sparse_array(G))

# Compute degree statistics
degrees = np.array([d for n, d in G.degree()])
k_mean = degrees.mean()
k2_mean = (degrees ** 2).mean()

# Plot degree distribution
plt.hist(degrees, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Degree Distribution of ER Network')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'degree_dist.png'))
plt.close()

# Save for appendices
with open(os.path.join(os.getcwd(), 'output', 'network_statistics.txt'), 'w') as f:
    f.write(f"Mean degree: {k_mean}\nSecond moment degree: {k2_mean}\nPopulation size: {N}\n")

# Return degree statistics
k_mean, k2_mean
import networkx as nx
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters for Barabasi-Albert network
N = 1000  # total population size (recommended for visualization and speed)
m = 2     # Each new node forms m edges (BA property)

# 1. Create BA network
G = nx.barabasi_albert_graph(N, m, seed=42)

# 2. Save to sparse npz file as required output format
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, "network.npz")
sparse.save_npz(network_path, nx.to_scipy_sparse_array(G))

# 3. Calculate mean degree <k> and second moment <k^2>
degrees = np.array([d for n, d in G.degree()])
mean_k = degrees.mean()
second_moment = (degrees ** 2).mean()

# 4. Save and plot degree distribution for visualization
plt.figure(figsize=(8,5))
plt.hist(degrees, bins=30, density=True, alpha=0.75, color="steelblue")
plt.xlabel("Degree")
plt.ylabel("Probability Density")
plt.title("Degree Distribution of Barabasi-Albert Network (N=1000, m=2)")
degree_plot_path = os.path.join(output_dir, "degree_distribution.png")
plt.savefig(degree_plot_path)
plt.close()

# Return the relevant variables for use in modeling
network_path, mean_k, second_moment, degree_plot_path
# We will construct two types of static networks frequently used in epidemic studies: Barabasi-Albert (scale-free) and Erdős-Rényi (random).
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import sparse

np.random.seed(42)
N = 1000  # Size of the population
# 1. Erdős-Rényi (ER) network
p_er = 0.01  # Probability for edge creation in ER
er_network = nx.erdos_renyi_graph(N, p_er)
# Save ER network
os.makedirs('output', exist_ok=True)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'ER_network.npz'), nx.to_scipy_sparse_array(er_network))
# Plot degree distribution for ER
plt.figure()
er_degrees = [d for n, d in er_network.degree()]
plt.hist(er_degrees, bins=30, alpha=0.7)
plt.title('Degree Distribution (ER)')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.savefig(os.path.join(os.getcwd(), 'output', 'ER_deg_dist.png'))
plt.close()
# Calculate network metrics for ER
mean_degree_er = np.mean(er_degrees)
second_moment_er = np.mean(np.array(er_degrees)**2)

# 2. Barabasi-Albert (BA) network
m_ba = 3  # Number of edges to attach from a new node (BA model)
ba_network = nx.barabasi_albert_graph(N, m_ba)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'BA_network.npz'), nx.to_scipy_sparse_array(ba_network))
# Plot degree distribution for BA
plt.figure()
ba_degrees = [d for n, d in ba_network.degree()]
plt.hist(ba_degrees, bins=30, alpha=0.7)
plt.title('Degree Distribution (BA)')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.savefig(os.path.join(os.getcwd(), 'output', 'BA_deg_dist.png'))
plt.close()
# Calculate network metrics for BA
mean_degree_ba = np.mean(ba_degrees)
second_moment_ba = np.mean(np.array(ba_degrees)**2)

metrics = {
    'ER': {'mean_degree': mean_degree_er, 'second_moment': second_moment_er},
    'BA': {'mean_degree': mean_degree_ba, 'second_moment': second_moment_ba},
}

import networkx as nx
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Parameters for both networks
N = 1000  # population size
average_degree = 8
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

# 1. Erdős-Rényi network (ER)
p_ER = average_degree / (N - 1)
G_ER = nx.erdos_renyi_graph(N, p_ER, seed=42)
csr_ER = nx.to_scipy_sparse_array(G_ER)
sparse.save_npz(os.path.join(output_dir, 'network-ER.npz'), csr_ER)

# 2. Barabási-Albert network (BA)
m_BA = average_degree // 2  # BA model parameter
G_BA = nx.barabasi_albert_graph(N, m_BA, seed=42)
csr_BA = nx.to_scipy_sparse_array(G_BA)
sparse.save_npz(os.path.join(output_dir, 'network-BA.npz'), csr_BA)

# --- Calculate mean degree and second moment
def network_stats(G):
    degrees = np.array([d for n, d in G.degree()])
    mean_k = degrees.mean()
    mean_k2 = (degrees ** 2).mean()
    return mean_k, mean_k2

# Degree stats
mean_k_ER, mean_k2_ER = network_stats(G_ER)
mean_k_BA, mean_k2_BA = network_stats(G_BA)

# --- Plots for understanding
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist([d for n, d in G_ER.degree()], bins=range(0, max(dict(G_ER.degree()).values())+2), alpha=0.8, color='skyblue')
plt.title('ER Degree Distribution')
plt.xlabel('Degree'); plt.ylabel('Count')

plt.subplot(1,2,2)
plt.hist([d for n, d in G_BA.degree()], bins=range(0, max(dict(G_BA.degree()).values())+2), alpha=0.8, color='orange')
plt.title('BA Degree Distribution')
plt.xlabel('Degree'); plt.ylabel('Count')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'degree-distributions.png'))

# Save degree statistics
degree_stats = {
    'mean_k_ER': mean_k_ER,
    'mean_k2_ER': mean_k2_ER,
    'mean_k_BA': mean_k_BA,
    'mean_k2_BA': mean_k2_BA
}

# Save reasoning information for appendices
reasoning_info = "Erdős-Rényi (ER) networks assume connections form at random with given probability, leading to Poisson degree distribution; Barabási-Albert (BA) builds using preferential attachment, resulting in a heavy-tailed (scale-free) degree distribution with a small number of highly-connected hubs. Both have the same average (8) for comparison, but BA has higher variance and hubs, impacting epidemic dynamics."


import numpy as np
import networkx as nx
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import os

# Configuration for reproducibility
np.random.seed(42)
# Parameters for network
N = 1000  # Total population size, a moderate size for demonstration
mean_degree = 8  # Reasonable average for close contact in communities
# For static human contact networks, an Erdős–Rényi (ER) network provides a reasonable baseline
p = mean_degree/(N-1)  # ER edge probability
# Generate network
G = nx.erdos_renyi_graph(N, p)
# Save network to output directory
os.makedirs(os.path.join(os.getcwd(), 'output'), exist_ok=True)
sparse.save_npz(os.path.join(os.getcwd(), "output", "network.npz"), nx.to_scipy_sparse_array(G))
# Calculate degree metrics
degrees = [d for n, d in G.degree()]
mean_deg = np.mean(degrees)
second_moment_deg = np.mean(np.array(degrees) **2)
# Plot degree distribution
plt.figure()
plt.hist(degrees, bins=20)
plt.title('Degree Distribution of ER Network (N=1000, <k>=8)')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.savefig(os.path.join(os.getcwd(), 'output', 'degree_distribution.png'))
plt.close()
# Output for reporting
network_details = {
    'mean_degree': mean_deg,
    'second_degree_moment': second_moment_deg,
}
network_path = os.path.join(os.getcwd(), "output", "network.npz")
plot_path = os.path.join(os.getcwd(), 'output', 'degree_distribution.png')
(network_details, network_path, plot_path)
import os
import numpy as np
import networkx as nx
from scipy import sparse
import matplotlib.pyplot as plt
# 1. Construct a static Erdős-Rényi (ER) network to represent the population.
N = 1000  # total population size
p = 0.015 # edge probability, chosen to get a realistic mean degree for epidemic modeling
G = nx.erdos_renyi_graph(N, p, seed=42)
# 2. Save the network to an .npz file for later simulation use
os.makedirs(os.path.join(os.getcwd(), 'output'), exist_ok=True)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network.npz'), nx.to_scipy_sparse_array(G))
# 3. Save degree distribution plot and network snapshot (optional for visualization)
degrees = [d for n, d in G.degree()]
plt.figure()
plt.hist(degrees, bins=30, color='blue', alpha=0.7)
plt.title('Degree Distribution of ER Network')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.tight_layout()
degree_hist_path = os.path.join(os.getcwd(), 'output', 'degree_hist.png')
plt.savefig(degree_hist_path)
plt.close()
# Calculate network metrics: mean degree <k> and <k^2>
deg_arr = np.array(degrees)
mean_k = deg_arr.mean()
mean_k2 = (deg_arr ** 2).mean()
(mean_k, mean_k2, degree_hist_path)
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import os

# Step 1: Network parameters (Barabasi-Albert scale-free network)
n = 1000       # 1000 nodes (typical for simulation scale, can adjust as necessary)
m = 4          # Each new node attaches to 4 existing nodes

# Step 2: Create network
G = nx.barabasi_albert_graph(n, m, seed=42)

# Step 3: Calculate statistics
k = np.array([d for n, d in G.degree()])
mean_k = k.mean()
second_moment_k = np.mean(k**2)

# Step 4: Save to disk
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
sparse.save_npz(os.path.join(output_dir, 'network.npz'), nx.to_scipy_sparse_array(G))

# Step 5: Plot and save degree distribution for visualization
plt.figure(figsize=(6,4))
degrees = [d for n, d in G.degree()]
plt.hist(degrees, bins=50, alpha=0.75)
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution (Barabasi-Albert, n=1000, m=4)')
plt.tight_layout()
degree_plot_path = os.path.join(output_dir, 'degree-distribution.png')
plt.savefig(degree_plot_path)
plt.close()

# Step 6: Save to record (network construction script for reproducibility)
with open(os.path.join(output_dir, 'network-construction.py'), 'w') as f:
    f.write('''"""
Barabasi-Albert scale-free network construction (n=1000, m=4)
"""
import networkx as nx
from scipy import sparse
import numpy as np
n = 1000
m = 4
G = nx.barabasi_albert_graph(n, m, seed=42)
sparse.save_npz('network.npz', nx.to_scipy_sparse_array(G))
''')

mean_k, second_moment_k, degree_plot_path

import numpy as np
import networkx as nx
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import os

# Parameters for network construction
N = 1000  # population size
avg_degree = 8  # representative from literature for social contact networks
p = avg_degree / (N-1)  # ER graph connection probability
# 1. Create Erdős-Rényi network
G = nx.erdos_renyi_graph(n=N, p=p, seed=42)
# 2. Save as sparse
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, "network.npz")
sparse.save_npz(network_path, nx.to_scipy_sparse_array(G))
# 3. Plot degree distribution
degrees = [d for n, d in G.degree()]
plt.figure()
plt.hist(degrees, bins=30, alpha=0.72)
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Degree Distribution (ER, N=1000)')
degree_hist_path = os.path.join(output_dir, "degree_dist.png")
plt.savefig(degree_hist_path)
plt.close()
# 4. Calculate moments
mean_degree = np.mean(degrees)
second_moment = np.mean(np.square(degrees))
# Save info
with open(os.path.join(output_dir, "network_info.txt"), "w") as f:
    f.write(f"Mean degree: {mean_degree}\nSecond moment: {second_moment}\n")
return_vars = ["network_path", "degree_hist_path", "mean_degree", "second_moment"]
# Chain of thought:
# 1. Select a moderate-sized, realistic population for static SIR simulation on a network. Recommend 1000 nodes for demonstration.
# 2. Use the Erdős–Rényi (ER) network for the contact structure, as this captures random mixing often assumed in SIR studies.
# 3. Set the ER network average degree to ~7 (well above connectivity threshold, appropriate for contagion processes).
# 4. Configure SIR parameters:
#    - Use beta = 0.4 (transmission rate per contact per time unit)
#    - Use gamma = 0.04 (recovery rate per time unit)
#    - But these are ODE values. For a network SIR, use beta_network = R0*gamma/q, with R0 = 2.5 as typical for COVID, q = (<k^2>-<k>)/<k>.
# 5. Create network, calculate <k>, <k^2>, q, and beta_network, save these and plot degree distribution.
# 6. Save network as sparse .npz.
import numpy as np
import networkx as nx
from scipy import sparse
import matplotlib.pyplot as plt
import os

np.random.seed(42)
N = 1000
mean_degree = 7
# ER model: p = mean_degree/(N-1)
p_edge = mean_degree/(N-1)
G = nx.erdos_renyi_graph(N, p_edge)
A = nx.to_scipy_sparse_array(G)

# Calculate degree moments
degree_seq = np.array([d for n, d in G.degree()])
mean_k = degree_seq.mean()
mean_k2 = (degree_seq**2).mean()
q = (mean_k2 - mean_k)/mean_k

# SIR rates
gamma = 0.04  # recovery rate
R0 = 2.5
# beta_network = R0*gamma/q
beta = R0*gamma/q

# Save network
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, "network.npz")
sparse.save_npz(network_path, A)
# Plot degree distribution
plt.figure()
plt.hist(degree_seq, bins=30, alpha=0.7)
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Degree distribution for ER network (N=1000, <k>=7)')
deg_fig_path = os.path.join(output_dir, "network_degree_dist.png")
plt.savefig(deg_fig_path)
plt.close()

# Return all needed variables
(mean_k, mean_k2, q, beta, gamma, network_path, deg_fig_path, N, degree_seq.tolist())
# 1. Network Construction: Erdős–Rényi (ER) Network as a canonical example
import networkx as nx
import numpy as np
import os
from scipy import sparse
import matplotlib.pyplot as plt

N = 1000  # population size
p = 0.01  # connection probability for ER network
network = nx.erdos_renyi_graph(N, p)
network_csr = nx.to_scipy_sparse_array(network)
network_degrees = np.array([d for n, d in network.degree()])
mean_degree = np.mean(network_degrees)
second_moment_degree = np.mean(network_degrees ** 2)

output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, 'network.npz')
sparse.save_npz(network_path, network_csr)

# Degree distribution plot
plt.figure()
plt.hist(network_degrees, bins=30, alpha=0.7)
plt.title('Degree distribution for ER network')
plt.xlabel('Degree')
plt.ylabel('Frequency')
deg_plot_path = os.path.join(output_dir, 'degree_dist.png')
plt.savefig(deg_plot_path)
plt.close()
# Export mean degree and second moment for next phase
mean_degree, second_moment_degree, network_path, deg_plot_path

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import sparse

# Parameters for Barabasi-Albert (Scale-Free) Network
total_population = 1000
mean_degree = 8
m_ba = mean_degree // 2  # Barabasi-Albert parameter

G = nx.barabasi_albert_graph(total_population, m_ba, seed=42)
# Save network to npz
if not os.path.exists('output'):
    os.makedirs('output')
csr = nx.to_scipy_sparse_array(G)
sparse.save_npz(os.path.join('output', 'network.npz'), csr)
# Compute degree statistics
degrees = np.array([d for n,d in G.degree()])
mean_deg = degrees.mean()
mean_deg2 = np.mean(degrees**2)
# Plot degree distribution
plt.figure(figsize=(8,6))
plt.hist(degrees, bins=30, density=True, color='skyblue', edgecolor='k')
plt.xlabel('Degree')
plt.ylabel('Fraction of Nodes')
plt.title('Degree Distribution (Barabasi-Albert)')
plt.tight_layout()
plt.savefig(os.path.join('output', 'degree_distribution.png'))
plt.close()

mean_deg, mean_deg2