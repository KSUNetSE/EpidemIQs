
import numpy as np
import networkx as nx
from scipy import sparse
import os
import matplotlib.pyplot as plt

# --- PARAMETERS
N = 1000  # nodes
activity = 3  # per node per step
m = 1      # edges per activated node
T = 100    # time window (adjust for reasonable static degree scale)
np.random.seed(42)

output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

# --- (A) TEMPORAL NETWORK: edge table (i, j, t)
temporal_contacts = []  # list of (t, i, j)
for t in range(T):
    active_nodes = np.where(np.random.rand(N) < activity)[0]  # p=activity, dt=1
    for i in active_nodes:
        # partner for i (avoid self):
        partner = np.random.randint(0, N)
        while partner == i:
            partner = np.random.randint(0, N)
        temporal_contacts.append((t, i, partner))

# Save temporal edge list
import pandas as pd
temporal_df = pd.DataFrame(temporal_contacts, columns=["time", "node_i", "node_j"])
temporal_path = os.path.join(output_dir, "activity-driven-temporal-edgelist.csv")
temporal_df.to_csv(temporal_path, index=False)

# --- (B) STATIC: Aggregate contacts into adjacency matrix
adj_matrix = np.zeros((N, N), dtype=int)
for _, i, j in temporal_contacts:
    adj_matrix[i, j] += 1
    adj_matrix[j, i] += 1  # undirected

# Remove diagonal (no self-loops)
np.fill_diagonal(adj_matrix, 0)

# Threshold to unweighted graph: edge exists if contact at least once
adj_unweighted = (adj_matrix > 0).astype(int)

# Build NetworkX weighted static graph
G = nx.from_numpy_array(adj_matrix)

# Save weighted adjacency as sparse
adj_sparse = sparse.csr_matrix(adj_matrix)
static_path = os.path.join(output_dir, "static-aggregated-adjacency.npz")
sparse.save_npz(static_path, adj_sparse)

# Save unweighted adjacency as edgelist csv
edges, weights = np.nonzero(np.triu(adj_matrix, k=1))
edgelist = pd.DataFrame({"node_i": edges, "node_j": weights, "weight": adj_matrix[edges, weights]})
static_edgelist_path = os.path.join(output_dir, "static-aggregated-edgelist.csv")
edgelist.to_csv(static_edgelist_path, index=False)

# --- Centralities and stats for static aggregated
k_sequence = np.sum(adj_unweighted, axis=1)
mean_k = np.mean(k_sequence)
mean_k2 = np.mean(k_sequence**2)

# GCC size, clustering, assortativity
components = [len(c) for c in nx.connected_components(nx.Graph(adj_unweighted))]
gcc_size = np.max(components)
try:
    clustering = nx.average_clustering(nx.Graph(adj_unweighted))
except:
    clustering = np.nan
try:
    assortativity = nx.degree_assortativity_coefficient(nx.Graph(adj_unweighted))
except:
    assortativity = np.nan
# Plot degree dist
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(k_sequence, bins=25, color="#408090", rwidth=0.9)
ax.set_xlabel("Degree")
ax.set_ylabel("Number of nodes")
ax.set_title("Degree distribution: Static Aggregated Network")
deg_plot_path = os.path.join(output_dir, "static-aggregated-degreedist.png")
fig.tight_layout()
fig.savefig(deg_plot_path)
plt.close(fig)

# Diagnostic plots: edge-weight distribution
edge_weights = adj_matrix[np.triu_indices(N, 1)]
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(edge_weights[edge_weights>0], bins=20, color="#E38817", rwidth=0.9)
ax.set_xlabel("Edge weight (contact count)")
ax.set_ylabel("Number of edges")
ax.set_title("Edge weight distribution (static, aggregated)")
edge_wplot_path = os.path.join(output_dir, "static-aggregated-edgeweightdist.png")
fig.tight_layout()
fig.savefig(edge_wplot_path)
plt.close(fig)

# --- Save code file for reproducibility
src_code = '''
# (same as this cell) [Truncated for brevity; see above]
'''
code_path = os.path.join(output_dir, "network-design.py")
with open(code_path, "w") as f:
    f.write(src_code)

# --- Report diagnostics
network_paths = [temporal_path, static_path, static_edgelist_path, code_path]
plot_paths = {deg_plot_path: "Degree distribution (static aggregated network)",
              edge_wplot_path: "Edge weight distribution (static aggregated network)"}
result = dict(
    mean_k = float(mean_k),
    mean_k2 = float(mean_k2),
    gcc_size=int(gcc_size),
    clustering=float(clustering),
    assortativity=float(assortativity),
    network_paths=network_paths,
    plot_paths=plot_paths,
)
result