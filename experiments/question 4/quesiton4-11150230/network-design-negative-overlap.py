
# Step 1: Parameters
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy import sparse

np.random.seed(42)

# Parameters for population and network size
N = 1000 # population size
mean_k_A = 8  # target mean degree for layer A (moderately high for rich epidemic)
mean_k_B = 8  # same for layer B for parity here
powerlaw_exp_A = 2.5  # heavy tail for layer A 
powerlaw_exp_B = 3.0  # slightly less heavy tail for B (flexibility for scenario)
hub_overlap_type = 'negative'  # choose from ['positive', 'none', 'negative'] per scenario guidance

# --- Helper functions for configuration model with given degree sequence ---
def generate_powerlaw_degree_sequence(N, exponent, mean_degree, min_deg=2, max_deg=None):
    '''Generates degree sequence for power-law distribution using the discrete method.'''
    if max_deg is None:
        max_deg = int(np.sqrt(N*mean_degree)*2)
    a = exponent
    # Degree values (integer range)
    k = np.arange(min_deg, max_deg+1)
    p_k = k**(-a)
    p_k = p_k / p_k.sum()
    degrees = np.random.choice(k, size=N, p=p_k)
    # Now rescale to get the right mean degree
    factor = mean_degree / degrees.mean()
    degrees = np.clip(np.round(degrees*factor), min_deg, max_deg).astype(int)
    # Make sum even
    if degrees.sum() % 2 == 1:
        idx = np.random.randint(N)
        degrees[idx] += 1
    return np.array(degrees)

# --- Step 3: Generate degree sequences ---
deg_seq_A = generate_powerlaw_degree_sequence(N, powerlaw_exp_A, mean_k_A)
deg_seq_B = generate_powerlaw_degree_sequence(N, powerlaw_exp_B, mean_k_B)

# --- Step 4: Impose inter-layer correlation/anti-correlation ---
if hub_overlap_type == 'positive':
    # match the ranking: sort by deg in A, assign to sorted in B
    idx_sort_A = np.argsort(-deg_seq_A)
    idx_sort_B = np.argsort(-deg_seq_B)
    deg_seq_B_sorted = deg_seq_B[idx_sort_B]
    deg_seq_B_new = np.zeros_like(deg_seq_B)
    deg_seq_B_new[idx_sort_A] = deg_seq_B_sorted
    deg_seq_B = deg_seq_B_new.copy()
elif hub_overlap_type == 'negative':
    # anti-align: A's largest to B's smallest
    idx_sort_A = np.argsort(-deg_seq_A)
    idx_sort_B = np.argsort(deg_seq_B)  # increasing order
    deg_seq_B_sorted = deg_seq_B[idx_sort_B]
    deg_seq_B_new = np.zeros_like(deg_seq_B)
    deg_seq_B_new[idx_sort_A] = deg_seq_B_sorted
    deg_seq_B = deg_seq_B_new.copy()
# else, leave as is for 'none' (random)

# --- Step 5: Build networks ---
# Use configuration model (simple graphs, remove parallel edges/self-loops)
G_A = nx.Graph(nx.configuration_model(deg_seq_A))
G_A.remove_edges_from(nx.selfloop_edges(G_A))
G_A = nx.Graph(G_A) # remove parallel edges
nx.set_node_attributes(G_A, {i: int(deg_seq_A[i]) for i in range(N)}, 'degree')

G_B = nx.Graph(nx.configuration_model(deg_seq_B))
G_B.remove_edges_from(nx.selfloop_edges(G_B))
G_B = nx.Graph(G_B)
nx.set_node_attributes(G_B, {i: int(deg_seq_B[i]) for i in range(N)}, 'degree')

# --- Step 6: Inter-layer degree correlation and hub overlap ---
degrees_A = np.array([G_A.degree(i) for i in G_A.nodes()])
degrees_B = np.array([G_B.degree(i) for i in G_B.nodes()])

# Pearson degree correlation
corr_AB = np.corrcoef(degrees_A, degrees_B)[0,1]

# Hub overlap index: fraction of top 10% nodes that are shared
percent_hub = 0.10
num_hub = int(N * percent_hub)
A_hubs = set(np.argsort(-degrees_A)[:num_hub])
B_hubs = set(np.argsort(-degrees_B)[:num_hub])
hub_overlap_fraction = len(A_hubs & B_hubs) / num_hub

# --- Step 7: Compute diagnostics per layer ---
def network_diagnostics(G):
    degree_sequence = [d for n, d in G.degree()]
    mean_k = np.mean(degree_sequence)
    k2 = np.mean(np.square(degree_sequence))
    # GCC size
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    gcc_size = len(components[0]) / N
    # Largest eigenvalue (spectral radius) of adjacency matrix
    A_sparse = nx.to_scipy_sparse_array(G, dtype=float)
    eigs = sparse.linalg.eigs(A_sparse, k=1, which='LR', return_eigenvectors=False)
    eig1 = float(np.real(eigs[0]))
    # Clustering coefficient
    clustering = nx.average_clustering(G)
    return dict(mean_k=mean_k, k2=k2, gcc_size=gcc_size, eig1=eig1, clustering=clustering)

info_A = network_diagnostics(G_A)
info_B = network_diagnostics(G_B)

# --- Step 8: Save layers, code, and plots ---
out_path = os.path.join(os.getcwd(), 'output')
os.makedirs(out_path, exist_ok=True)
sparse.save_npz(os.path.join(out_path, 'network-layerA.npz'), nx.to_scipy_sparse_array(G_A))
sparse.save_npz(os.path.join(out_path, 'network-layerB.npz'), nx.to_scipy_sparse_array(G_B))

with open(os.path.join(out_path, 'network-design-negative-overlap.py'), 'w') as f:
    f.write('# Code for constructing negative overlap multiplex network\n')
    f.write(open(__file__).read() if '__file__' in globals() else '[[This code block]]\n')

def save_plot(fig, name, caption):
    path = os.path.join(out_path, name)
    fig.savefig(path)
    plt.close(fig)
    return path, caption

# Degree distribution plot A
fig, ax = plt.subplots()
ax.hist(degrees_A, bins=30, color='royalblue', alpha=0.7)
ax.set_xlabel('Degree (Layer A)')
ax.set_ylabel('Count')
ax.set_title('Degree Distribution: Layer A')
degdistA_path, degdistA_cap = save_plot(fig, 'degree-distribution-layerA.png', 'Degree distribution for layer A (power-law, heavy tail, anti-overlapped with B).')

# Degree distribution plot B
fig, ax = plt.subplots()
ax.hist(degrees_B, bins=30, color='tomato', alpha=0.7)
ax.set_xlabel('Degree (Layer B)')
ax.set_ylabel('Count')
ax.set_title('Degree Distribution: Layer B')
degdistB_path, degdistB_cap = save_plot(fig, 'degree-distribution-layerB.png', 'Degree distribution for layer B (slightly less heavy tail, anti-overlapped with A).')

# Degree correlation scatter
fig, ax = plt.subplots()
ax.scatter(degrees_A, degrees_B, alpha=0.15, s=5)
ax.set_xlabel('Degree (Layer A)')
ax.set_ylabel('Degree (Layer B)')
ax.set_title(f'Node degree correlation: Pearson r={corr_AB:.2f}')
degcorr_path, degcorr_cap = save_plot(fig, 'degree-correlation-scatter.png', f'Degree correlation scatter plot between layers (anti-correlation, r={corr_AB:.2f})')

# Hub overlap bar
fig, ax = plt.subplots()
ax.bar(['Hub Overlap'], [hub_overlap_fraction], color='purple')
ax.set_ylim(0, 1)
ax.set_ylabel('Fraction')
ax.set_title('Fraction of shared top 10% degree hubs')
huboverlap_path, huboverlap_cap = save_plot(fig, 'hub-overlap-bar.png', 'Fraction of nodes that are hubs (top 10% degree) in both layers.')

# --- Return variable dict (for use in summary) ---
paths = dict(
    layerA='output/network-layerA.npz',
    layerB='output/network-layerB.npz',
    code='output/network-design-negative-overlap.py',
    degdistA=degdistA_path,
    degdistB=degdistB_path,
    degcorr=degcorr_path,
    huboverlap=huboverlap_path
)

summaries = dict(
    info_A=info_A,
    info_B=info_B,
    corr_AB=corr_AB,
    hub_overlap_fraction=hub_overlap_fraction,
    N=N
)
