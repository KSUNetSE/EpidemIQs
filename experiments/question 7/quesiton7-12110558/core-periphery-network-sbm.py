
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import seaborn as sns

# PARAMETERS
N = 100         # Total banks (flexible, tractable for demo)
kc = 0.9        # Core-core probability (user can change for test)
kp = 0.075      # Periphery-periphery probability
kcp = 0.3       # Core-Periphery probability
rng_seed = 42
np.random.seed(rng_seed)

Nc = int(0.1 * N)
Np = N - Nc
sizes = [Nc, Np]
labels = ["core"] * Nc + ["periphery"] * Np
p_matrix = np.array([[kc, kcp], [kcp, kp]])

# CORE-PERIPHERY SBM NETWORK CONSTRUCTION
G = nx.stochastic_block_model(sizes, p_matrix, seed=rng_seed, directed=False)

# Annotate node group
for idx, group in enumerate(labels):
    G.nodes[idx]['group'] = group

# SAVE NETWORK (sparse adj matrix)
output_dir = os.path.join(os.getcwd(),"output")
os.makedirs(output_dir, exist_ok=True)
network_path = os.path.join(output_dir, f"core-periphery-network-kc{kc:.2f}.npz")
sparse.save_npz(network_path, nx.to_scipy_sparse_array(G))

# SAVE GROUP LABELS (as .csv for analysis)
group_path = os.path.join(output_dir, f"core-periphery-groups-kc{kc:.2f}.csv")
with open(group_path, 'w') as f:
    f.write("node,group\n")
    for i, node in enumerate(G.nodes()):
        f.write(f"{node},{labels[i]}\n")

# METRICS
k_list = np.array([d for n, d in G.degree()])
mean_deg = float(np.mean(k_list))
second_moment = float(np.mean(k_list ** 2))
largest_cc = len(max(nx.connected_components(G), key=len))
core_degrees = k_list[:Nc]
periphery_degrees = k_list[Nc:]
mean_core = float(np.mean(core_degrees))
mean_periphery = float(np.mean(periphery_degrees))

# Degree assortativity (only if network is not too small/disconnected)
try:
    degree_assort = float(nx.degree_assortativity_coefficient(G))
except Exception:
    degree_assort = None

# PLOTS: DEGREE DISTRIBUTION
fig, ax = plt.subplots(figsize=(7,4))
sns.histplot(core_degrees, bins=range(0, max(core_degrees) + 2), color="tab:red", alpha=0.7, label="Core", stat="probability")
sns.histplot(periphery_degrees, bins=range(0, max(periphery_degrees) + 2), color="tab:blue", alpha=0.5, label="Periphery", stat="probability")
plt.legend()
plt.xlabel("Degree k")
plt.ylabel("Fraction of nodes")
plt.title("Degree Distribution by Block (Core & Periphery)")
degree_plot_path = os.path.join(output_dir, f"degree-distribution-core-periphery-kc{kc:.2f}.png")
plt.tight_layout()
plt.savefig(degree_plot_path)
plt.close()

# PLOT: NETWORK LAYOUT (Spring, color by group)
core_color = "#d62728"
periph_color = "#1f77b4"
colors = [core_color if lbl=="core" else periph_color for lbl in labels]
pos = nx.spring_layout(G, seed=24)
plt.figure(figsize=(6,6))
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=80, alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.35, width=0.5)
plt.axis('off')
plt.title("Core-Periphery SBM: Network Visualization (kc=%.2f)" % kc)
network_viz_path = os.path.join(output_dir, f"network-visualization-core-periphery-kc{kc:.2f}.png")
plt.tight_layout()
plt.savefig(network_viz_path)
plt.close()

return_vars = [
    'network_path', 'degree_plot_path', 'network_viz_path', 'mean_deg', 'second_moment',
    'largest_cc', 'mean_core', 'mean_periphery', 'degree_assort', 'Nc', 'Np', 'kp', 'kcp', 'kc', 'group_path'
]
