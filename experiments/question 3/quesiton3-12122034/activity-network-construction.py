
import numpy as np
import networkx as nx
from scipy import sparse
import matplotlib.pyplot as plt
import os

# Parameters
total_nodes = 1000
alpha = 0.1
m = 2
T = 1000  # observation window for aggregation

gamma = 1.0  # recovery rate
desired_avg_degree = 2 * m * alpha  # should be 0.4
expected_total_activations = int(total_nodes * alpha * T)  # expected total node activations

# Initialize holders
# For temporal: record (time, src, dst) triples
edge_times = []

# For aggregation: undirected pairwise counts
from collections import defaultdict
agg_edge_weights = defaultdict(int)

rng = np.random.default_rng(seed=42)  # reproducibility

for t in range(T):
    # Which nodes are active this timestep?
    activations = rng.uniform(size=total_nodes) < alpha
    active_nodes = np.where(activations)[0]
    # For each active node, sample m other nodes at random (no self-loops)
    for node in active_nodes:
        possible_partners = [i for i in range(total_nodes) if i != node]
        partners = rng.choice(possible_partners, size=m, replace=False)
        for partner in partners:
            edge = tuple(sorted((node, partner)))
            agg_edge_weights[edge] += 1
            edge_times.append((t, node, partner))

#--------------------------
# Create aggregated static network
G = nx.Graph()
G.add_nodes_from(range(total_nodes))
for (i, j), w in agg_edge_weights.items():
    # Edge weight represents total number of contacts (here, typically ~1)
    G.add_edge(i, j, weight=w)

#---------------------------
# Save networks
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
# Static network (as sparse for fast use)
static_path = os.path.join(output_dir, "activity-aggregated-network.npz")
sparse.save_npz(static_path, nx.to_scipy_sparse_array(G))
# Save weights separately
data_w = np.zeros((G.number_of_edges(), 3), dtype=int)
for idx, (i, j, d) in enumerate(G.edges(data=True)):
    data_w[idx] = [i, j, d['weight']]
np.savez_compressed(os.path.join(output_dir, "activity-aggregated-weights.npz"), edges=data_w)

# Temporal edge list (as npz)
edge_time_arr = np.array(edge_times, dtype=int)
temporal_path = os.path.join(output_dir, "activity-temporal-contacts.npz")
np.savez_compressed(temporal_path, edge_times=edge_time_arr)

#----------------------------
# Diagnostics and plots for static network
# Degree sequence
deg_seq = np.array([d for n, d in G.degree()])
mean_k = deg_seq.mean()
mean_k2 = (deg_seq **2).mean()

plt.figure()
plt.hist(deg_seq, bins=np.arange(deg_seq.max()+2)-0.5, density=True, color='steelblue')
plt.xlabel('Degree')
plt.ylabel('Fraction of nodes')
plt.title('Degree distribution (aggregated network)')
deg_hist_path = os.path.join(output_dir,"activity-degree-distribution.png")
plt.savefig(deg_hist_path)
plt.close()

# Distribution of edge weights (for aggregated network)
all_weights = np.array([d['weight'] for i, j, d in G.edges(data=True)])
plt.figure()
plt.hist(all_weights, bins=np.arange(all_weights.max()+2)-0.5, density=True, color='limegreen')
plt.xlabel('Edge weight (number of contacts)')
plt.ylabel('Fraction of links')
plt.title('Edge weight distribution (aggregated network)')
weight_hist_path = os.path.join(output_dir,"activity-weight-distribution.png")
plt.savefig(weight_hist_path)
plt.close()

# Component size distribution
components = [len(c) for c in nx.connected_components(G)]
plt.figure()
plt.hist(components, bins=np.arange(max(components)+2)-0.5, color='firebrick', density=True)
plt.xlabel('Component size')
plt.ylabel('Fraction of components')
plt.title('Component size distribution (aggregated network)')
comp_hist_path = os.path.join(output_dir,"activity-component-size-distribution.png")
plt.savefig(comp_hist_path)
plt.close()

# Main diagnostics output
GCC_size = max(components)
prop_in_GCC = GCC_size / total_nodes

# For temporal: edge count per node (as initiator+as recipient per T)
contact_counts = np.zeros(total_nodes, dtype=int)
for t, i, j in edge_times:
    contact_counts[i] += 1
    contact_counts[j] += 1

plt.figure()
plt.hist(contact_counts, bins=30, color='orange', density=True)
plt.xlabel('Total contacts per node (T steps)')
plt.ylabel('Fraction of nodes')
plt.title('Contact count distribution (temporal network)')
temp_contacts_path = os.path.join(output_dir, "activity-temporal-contactcount.png")
plt.savefig(temp_contacts_path)
plt.close()

# Return diagnostics
{
    'static_network_path': static_path,
    'temporal_network_path': temporal_path,
    'degree_hist_path': deg_hist_path,
    'weight_hist_path': weight_hist_path,
    'comp_hist_path': comp_hist_path,
    'mean_k': mean_k,
    'mean_k2': mean_k2,
    'gcc_size': GCC_size,
    'gcc_frac': prop_in_GCC,
    'temporal_contacts_path': temp_contacts_path
}