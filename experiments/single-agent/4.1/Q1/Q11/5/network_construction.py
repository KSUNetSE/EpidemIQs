
import networkx as nx
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
import os

# Construct two networks: 
# 1) Homogeneous-mixing (Erdős-Rényi random graph with narrow degree)
# 2) Degree-heterogeneous (Barabási-Albert scale-free)

N = 1000
# 1. ER graph with mean degree k = 8
kmean_er = 8
p_er = kmean_er / (N-1)
G_er = nx.erdos_renyi_graph(N, p_er, seed=42)

# 2. Barabási-Albert scale-free with mean degree ~ 8
G_ba = nx.barabasi_albert_graph(N, int(kmean_er/2), seed=42)

# Save networks
outdir = os.path.join(os.getcwd(), 'output')
if not os.path.exists(outdir):
    os.makedirs(outdir)
sparse.save_npz(os.path.join(outdir, "network-er.npz"), nx.to_scipy_sparse_array(G_er))
sparse.save_npz(os.path.join(outdir, "network-ba.npz"), nx.to_scipy_sparse_array(G_ba))

# Calculate <k> and <k^2>
def degree_moments(G):
    degrees = np.array([d for n, d in G.degree()])
    kmean = np.mean(degrees)
    k2 = np.mean(degrees**2)
    return kmean, k2
kmean_er, k2_er = degree_moments(G_er)
kmean_ba, k2_ba = degree_moments(G_ba)

# Plot degree distributions
plt.figure(figsize=(8,4))
plt.hist([d for n,d in G_er.degree()], bins=range(0, max(dict(G_er.degree()).values())+1), alpha=0.6, label="ER")
plt.hist([d for n,d in G_ba.degree()], bins=range(0, max(dict(G_ba.degree()).values())+1), alpha=0.6, label="BA")
plt.xlabel('Degree')
plt.ylabel('Count')
plt.legend()
plt.title('Degree Distributions: ER vs BA')
plt.tight_layout()
plt.savefig(os.path.join(outdir, "degree-dist-er-vs-ba.png"))

paths = {
    "er": os.path.join(outdir, "network-er.npz"),
    "ba": os.path.join(outdir, "network-ba.npz"),
    "plot": os.path.join(outdir, "degree-dist-er-vs-ba.png")
}
(kmean_er, k2_er), (kmean_ba, k2_ba), paths
# Chain of Thought:
# 1. We need to model two scenarios: (a) SEIR on a homogeneous-mixing (fully connected) network and (b) SEIR on a degree-heterogeneous (scale-free) network.
# 2. The network for homogeneous-mixing will be an Erdos-Renyi random graph with high connection probability to mimic well-mixed behavior.
# 3. The degree-heterogeneous network will be constructed using the Barabasi-Albert scale-free model.
# 4. Both networks will have the same number of nodes (N=2000) and comparable mean degree.
# 5. We will save both network structures for subsequent simulations.
# 6. We'll compute and report the mean degree <k> and the second moment <k^2>.
# 7. Save the constructed networks and plots of their degree distributions.
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import sparse

np.random.seed(42)
N = 2000
mean_degree = 12  # Chosen for moderately connected, realistic sized network

# Homogeneous-mixing: ER network
p_ER = mean_degree / (N - 1)
G_ER = nx.erdos_renyi_graph(N, p_ER)
ER_adjacency = nx.to_scipy_sparse_array(G_ER)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network-ER.npz'), ER_adjacency)

# Degree-heterogeneous: Scale-free (Barabasi-Albert) network
m_BA = mean_degree // 2  # BA parameter: new node connects to m_BA existing nodes
graph_BA = nx.barabasi_albert_graph(N, m_BA)
BA_adjacency = nx.to_scipy_sparse_array(graph_BA)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network-BA.npz'), BA_adjacency)

# Compute network statistics
ER_degrees = np.array([deg for n,deg in G_ER.degree()])
BA_degrees = np.array([deg for n,deg in graph_BA.degree()])

ER_mean_degree = ER_degrees.mean()
BA_mean_degree = BA_degrees.mean()
ER_k2 = (ER_degrees**2).mean()
BA_k2 = (BA_degrees**2).mean()

# Plot degree distributions and save
plt.figure(figsize=(8,5))
plt.hist(ER_degrees, bins=40, color='blue', edgecolor='black')
plt.title('Erdos-Renyi Network Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.savefig(os.path.join(os.getcwd(), 'output', 'ER_degree_dist.png'))
plt.close()

plt.figure(figsize=(8,5))
plt.hist(BA_degrees, bins=40, color='red', edgecolor='black')
plt.title('Barabasi-Albert (Scale-Free) Network Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.savefig(os.path.join(os.getcwd(), 'output', 'BA_degree_dist.png'))
plt.close()

results = {
    'ER_network_path': os.path.join(os.getcwd(), 'output', 'network-ER.npz'),
    'BA_network_path': os.path.join(os.getcwd(), 'output', 'network-BA.npz'),
    'ER_degree_mean': ER_mean_degree,
    'ER_degree_2nd_moment': ER_k2,
    'BA_degree_mean': BA_mean_degree,
    'BA_degree_2nd_moment': BA_k2,
    'ER_degree_plot': os.path.join(os.getcwd(), 'output', 'ER_degree_dist.png'),
    'BA_degree_plot': os.path.join(os.getcwd(), 'output', 'BA_degree_dist.png'),
}
results
import networkx as nx
import numpy as np
import os
import scipy.sparse as sparse
import matplotlib.pyplot as plt

output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

n = 1000  # Number of nodes
k_hom = 10  # Regular degree

# Corrected: In random_regular_graph, parameter order is d=degree, n=num_nodes
G_hom = nx.random_regular_graph(d=k_hom, n=n, seed=42)
A_hom = nx.to_scipy_sparse_array(G_hom)
sparse.save_npz(os.path.join(output_dir, 'network_homogeneous.npz'), A_hom)

degree_hom = np.array([d for _, d in G_hom.degree()])
mean_deg_hom = np.mean(degree_hom)
mean_deg2_hom = np.mean(degree_hom**2)

# Barabasi-Albert network for heterogeneous case:
m_ba = 5
G_ba = nx.barabasi_albert_graph(n=n, m=m_ba, seed=24)
A_ba = nx.to_scipy_sparse_array(G_ba)
sparse.save_npz(os.path.join(output_dir, 'network_heterogeneous.npz'), A_ba)

degree_ba = np.array([d for _, d in G_ba.degree()])
mean_deg_ba = np.mean(degree_ba)
mean_deg2_ba = np.mean(degree_ba**2)

plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.hist(degree_hom, bins=np.arange(k_hom-2, k_hom+3), color='blue', alpha=0.7)
plt.title('Homogeneous Network Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.subplot(1,2,2)
plt.hist(degree_ba, bins=30, color='orange', alpha=0.7)
plt.title('Heterogeneous (BA) Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.tight_layout()
deg_out_path = os.path.join(output_dir, 'degree_distributions.png')
plt.savefig(deg_out_path)
plt.close()

network_details = {
    'homogeneous': {'mean_degree': mean_deg_hom, 'second_moment': mean_deg2_hom},
    'heterogeneous': {'mean_degree': mean_deg_ba, 'second_moment': mean_deg2_ba}
}

deg_out_path, network_details

# Correction: Use explicit power-law degree sequence via numpy.
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import os

os.makedirs('output', exist_ok=True)
N = 2500
mean_degree = 8  # try to match means for fair comparison
# Homogeneous (ER)
p_er = mean_degree / (N - 1)
G_er = nx.erdos_renyi_graph(N, p_er, seed=42)
A_er = nx.to_scipy_sparse_array(G_er)
# Degree-heterogeneous (power-law)
min_deg = 2
pl_exponent = 2.7
max_deg = 50
np.random.seed(100)
raw_seq = np.random.zipf(a=pl_exponent, size=N)
seq = [min(max(min_deg, k), max_deg) for k in raw_seq]  # clamp
# Sum must be even for Configuration
if sum(seq) % 2 != 0:
    seq[0] += 1
G_conf = nx.configuration_model(seq, seed=43)
G_conf = nx.Graph(G_conf)  # Remove parallel edges
G_conf.remove_edges_from(nx.selfloop_edges(G_conf))
A_conf = nx.to_scipy_sparse_array(G_conf)
# Save networks
sparse.save_npz('output/network-er.npz', A_er)
sparse.save_npz('output/network-conf.npz', A_conf)
# Plot degree distributions
plt.figure(figsize=(7,4))
plt.hist([d for n, d in G_er.degree()], bins=40, alpha=0.7, label='ER')
plt.hist([d for n, d in G_conf.degree()], bins=range(1,55), alpha=0.7, label='Config PL')
plt.xlabel('Degree'); plt.ylabel('Nodes'); plt.title('Degree Distribution (log y)')
plt.yscale('log')
plt.legend(); plt.tight_layout();
plt.savefig('output/degree-dist.png')
plt.close()
# Calculate mean/2nd moment
er_deg = np.array([d for n, d in G_er.degree()])
er_k1 = er_deg.mean()
er_k2 = (er_deg**2).mean()
conf_deg = np.array([d for n, d in G_conf.degree()])
conf_k1 = conf_deg.mean()
conf_k2 = (conf_deg**2).mean()
results = {
    'er_k1': float(er_k1),
    'er_k2': float(er_k2),
    'conf_k1': float(conf_k1),
    'conf_k2': float(conf_k2),
    'net_paths': ['output/network-er.npz', 'output/network-conf.npz'],
    'plot_path': 'output/degree-dist.png',
    'N': N
}
results
# --- Step 1: For both types of networks, we need the mean degree <k> and the degree second moment <k^2> 
# Let's define network parameters for both a homogeneous-mixing (Erdos-Renyi) and a heterogeneous (Barabasi-Albert) network.
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import os

# Parameters
N = 1000  # population size
mean_degree = 12  # same average degree for both networks (typical for contact networks)
# Homogeneous-mixing (Erdos-Renyi)
p_er = mean_degree/(N-1)
G_er = nx.erdos_renyi_graph(n=N, p=p_er)
# Heterogeneous (Barabasi-Albert)
m_ba = mean_degree // 2  # m parameter, BA mean degree ~ 2m
G_ba = nx.barabasi_albert_graph(n=N, m=m_ba)

# Save networks
os.makedirs(os.path.join(os.getcwd(), 'output'), exist_ok=True)
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network-er.npz'), nx.to_scipy_sparse_array(G_er))
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network-ba.npz'), nx.to_scipy_sparse_array(G_ba))

# Degree stats
mean_degree_er = np.mean([d for n, d in G_er.degree()])
mean_degree2_er = np.mean([d**2 for n, d in G_er.degree()])
mean_degree_ba = np.mean([d for n, d in G_ba.degree()])
mean_degree2_ba = np.mean([d**2 for n, d in G_ba.degree()])

# Save plots
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.hist([d for n,d in G_er.degree()], bins=40, alpha=0.7, label='Erdos-Renyi')
plt.hist([d for n,d in G_ba.degree()], bins=40, alpha=0.7, label='Barabasi-Albert')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'degree-dist.png'))
plt.close()

# Return paths and degree stats
result = {
    'network_paths': [os.path.join(os.getcwd(), 'output', 'network-er.npz'), os.path.join(os.getcwd(), 'output', 'network-ba.npz')],
    'degree_stats': {
        'er': {'mean': mean_degree_er, 'second_moment': mean_degree2_er},
        'ba': {'mean': mean_degree_ba, 'second_moment': mean_degree2_ba}
    },
    'plot_paths': {'degree-dist.png': 'Degree distributions of Erdos-Renyi (homogeneous) and Barabasi-Albert (heterogeneous) networks'}
}
result