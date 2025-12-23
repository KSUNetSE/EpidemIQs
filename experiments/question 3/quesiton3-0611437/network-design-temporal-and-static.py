
# --- PARAMETERS AND ACTIVITY DISTRIBUTION SETUP ---
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import random
np.random.seed(42)
random.seed(42)

# Simulation params
N = 1000                                 # population size
m = 1                                    # contacts per activation
duration = 1000                          # time steps for aggregation
activity_dist = 'pareto'                 # use a plausible heavy-tailed activity dist
pareto_alpha = 2.5                       # shape parameter (can experiment)
activity_min = 0.01                      # min activity rate (avoid 0)

# 1. Generate node activities
if activity_dist == 'pareto':
    a_raw = (np.random.pareto(pareto_alpha, N) + 1) * activity_min
else:
    a_raw = np.random.uniform(activity_min, 1, N)
# Cap rates at 1
activity = np.clip(a_raw, activity_min, 1)
# Node label: activity dict
node_activity = dict((i, activity[i]) for i in range(N))

# Save activity for reference/permanence
activity_path = os.path.join(os.getcwd(), 'output', 'activity_data.npy')
os.makedirs(os.path.dirname(activity_path), exist_ok=True)
np.save(activity_path, activity)

# --- TEMPORAL ACTIVITY-DRIVEN NETWORK: EDGE LIST GENERATION ---
temporal_edges = []  # list of (t, i, j)
for t in range(duration):
    active_nodes = [i for i in range(N) if np.random.rand() < activity[i]]
    for i in active_nodes:
        # choose m unique partners, no self-loop, per time step
        partners = set()
        while len(partners) < m:
            partner = random.randint(0, N-1)
            if partner != i:
                partners.add(partner)
        for j in partners:
            edge = (t, min(i,j), max(i,j)) # undirected, canonicalize (i<j)
            temporal_edges.append(edge)

# Save temporal edge list
edge_t_path = os.path.join(os.getcwd(), 'output', 'temporal-contact-edges.csv')
with open(edge_t_path, 'w') as f:
    f.write('t,i,j\n')
    for t, i, j in temporal_edges:
        f.write(f'{t},{i},{j}\n')

# --- AGGREGATED STATIC WEIGHTED NETWORK CONSTRUCTION ---
from collections import defaultdict
edge_weights = defaultdict(int)
for t, i, j in temporal_edges:
    edge_weights[(i, j)] += 1

# Build weighted network
G_static = nx.Graph()
G_static.add_nodes_from(range(N))
for (i, j), w in edge_weights.items():
    G_static.add_edge(i, j, weight=w)

static_network_path = os.path.join(os.getcwd(), 'output', 'static-weighted-network.npz')
sparse.save_npz(static_network_path, nx.to_scipy_sparse_array(G_static, weight='weight'))

# --- PLOTS / DIAGNOSTIC METRICS ---
# Plot activity distribution
plt.figure()
plt.hist(activity, bins=50, color='skyblue', edgecolor='k', log=True)
act_plot_path = os.path.join(os.getcwd(), 'output', 'activity-histogram.png')
plt.xlabel('Activity Rate a'); plt.ylabel('Count (log)'); plt.title('Node Activity Distribution')
plt.savefig(act_plot_path, bbox_inches='tight'); plt.close()

# Static weighted network: plot degree distribution
degrees = np.array([G_static.degree(i) for i in G_static.nodes])
plt.figure()
plt.hist(degrees, bins=40, color='orange', edgecolor='k', log=True)
deg_plot_path = os.path.join(os.getcwd(), 'output', 'static-degree-histogram.png')
plt.xlabel('Degree'); plt.ylabel('Count (log)'); plt.title('Degree Dist. (Aggregated Network)')
plt.savefig(deg_plot_path, bbox_inches='tight'); plt.close()

# Plot edge weight distribution
weights = np.array([d['weight'] for u,v,d in G_static.edges(data=True)])
plt.figure()
plt.hist(weights, bins=40, color='mediumseagreen', edgecolor='k', log=True)
weight_plot_path = os.path.join(os.getcwd(), 'output', 'edge-weight-histogram.png')
plt.xlabel('Edge Weight'); plt.ylabel('Count (log)'); plt.title('Edge Weight Dist. (Aggregated Network)')
plt.savefig(weight_plot_path, bbox_inches='tight'); plt.close()

# --- NETWORK METRICS ---
# Mean degree and second moment for static network
mean_deg = degrees.mean()
mean_deg2 = (degrees**2).mean()

# Temporal (mean degree per time step = 2*m*(expected #active per t)/N)
avg_active_per_t = activity.sum()
avg_temporal_degree_per_timestep = m * avg_active_per_t / N * 2  # every edge reaches two nodes

# Second moment for temporal network (per time step, approx. Poisson)
temp_deg_list = [0]*N
for t, i, j in temporal_edges:
    temp_deg_list[i] += 1
    temp_deg_list[j] += 1
mean_deg_temporal = np.mean(temp_deg_list) / duration
mean_deg2_temporal = np.mean((np.array(temp_deg_list)/duration)**2)

# Save short summary/metrics
diag_path = os.path.join(os.getcwd(), 'output', 'network-diagnostics.txt')
with open(diag_path, 'w') as f:
    f.write(f'Activity mean: {np.mean(activity):.4f}, Activity 2nd moment: {np.mean(activity**2):.6f}\n')
    f.write(f'Temporal network: mean degree/timestep: {avg_temporal_degree_per_timestep:.3f}, 2nd moment: {mean_deg2_temporal:.5f}\n')
    f.write(f'Static network: mean degree: {mean_deg:.3f}, 2nd moment: {mean_deg2:.5f}, GCC size: {len(max(nx.connected_components(G_static), key=len))}\n')
    if len(weights) > 0:
        f.write(f'Edge weight mean: {np.mean(weights):.3f}, median: {np.median(weights)}, max: {np.max(weights)}, min: {np.min(weights)}\n')
    f.write(f'Assortativity (degree): {nx.degree_assortativity_coefficient(G_static):.5f}\n')
    try:
        gcc_nodes = max(nx.connected_components(G_static), key=len)
        gcc = G_static.subgraph(gcc_nodes)
        f.write(f'Average shortest path (GCC): {nx.average_shortest_path_length(gcc):.3f}\n')
    except Exception as e:
        f.write('Average shortest path (GCC): Error '+str(e)+'\n')
    f.write(f'Global clustering coefficient: {nx.transitivity(G_static):.5f}\n')

diag = {
    'mean_deg': mean_deg,
    'mean_deg2': mean_deg2,
    'mean_deg_temporal': mean_deg_temporal,
    'mean_deg2_temporal': mean_deg2_temporal,
    'mean_activity': np.mean(activity),
    'act2': np.mean(activity**2),
    'number_nodes': N,
    'gcc_size': len(max(nx.connected_components(G_static), key=len)),
    'number_of_edges': G_static.number_of_edges(),
    'mean_edge_weight': np.mean(weights) if len(weights) > 0 else 0
}

diag, [activity_path, edge_t_path, static_network_path, diag_path, act_plot_path, deg_plot_path, weight_plot_path]