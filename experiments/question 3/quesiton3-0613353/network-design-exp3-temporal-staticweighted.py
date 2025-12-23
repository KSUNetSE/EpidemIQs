
# PARAMETERS AND ACTIVITY GENERATION
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from scipy import sparse
from collections import defaultdict
np.random.seed(44)
random.seed(44)

# Population and network parameters
N = 10000
m = 2
T = 2000                               # aggregation window for robust statistics
activity_mean_target = 0.1
activity_dist = 'powerlaw'
gamma = 2.5
activity_min = 0.01
activity_max = 1.0

# -- Generate power-law activities so that <a>~0.1
# sample activities first, then rescale to get exact mean
r = np.random.rand(N)
exponent = 1 - gamma
amin_exp = activity_min ** exponent
amax_exp = activity_max ** exponent
activity_raw = (r * (amax_exp - amin_exp) + amin_exp) ** (1 / exponent)
activity_raw = np.clip(activity_raw, activity_min, activity_max)
# Normalize activities so mean = 0.1
activity = activity_raw * (activity_mean_target / activity_raw.mean())
activity = np.clip(activity, activity_min, activity_max)
node_activity = dict((i, activity[i]) for i in range(N))

# Save node activities
activity_path = os.path.join(os.getcwd(), 'output', 'exp3-powerlaw-activity.npy')
os.makedirs(os.path.dirname(activity_path), exist_ok=True)
np.save(activity_path, activity)

# TEMPORAL CONTACTS: Build ephemeral (t,i,j) list
edge_t_path = os.path.join(os.getcwd(), 'output', 'exp3-temporal-contact-edges.csv')
temporal_edges = []
contacts_count = defaultdict(int) # for static aggregation
for t in range(T):
    active_nodes = [i for i in range(N) if np.random.rand() < activity[i]]
    for i in active_nodes:
        partners = set()
        while len(partners) < m:
            j = random.randint(0, N-1)
            if j != i:
                partners.add(j)
        for j in partners:
            eid = (min(i,j), max(i,j))
            temporal_edges.append((t, i, j))
            contacts_count[eid] += 1
with open(edge_t_path, 'w') as f:
    f.write('t,i,j\n')
    for t, i, j in temporal_edges:
        f.write(f'{t},{i},{j}\n')

# STATIC WEIGHTED NETWORK: Edge weights are frequency / T
G_static_w = nx.Graph()
G_static_w.add_nodes_from(range(N))
for (i, j), cnt in contacts_count.items():
    G_static_w.add_edge(i, j, weight=cnt/T)
static_network_path = os.path.join(os.getcwd(), 'output', 'exp3-static-weighted.npz')
sparse.save_npz(static_network_path, nx.to_scipy_sparse_array(G_static_w, weight='weight'))

# DIAGNOSTICS: Degree, strength, mean and 2nd degree moment
strengths = np.array([sum(d['weight'] for _, _, d in G_static_w.edges(n, data=True)) for n in range(N)])
degrees = np.array([G_static_w.degree(n) for n in range(N)])
mean_deg = degrees.mean()
mean_deg2 = (degrees**2).mean()

# For activities
a2 = (activity**2).mean()

# For temporal: per-timestep mean degree (average contacts per node per step)
temp_deg_per_t = np.zeros(T)
for t in range(T):
    count_t = sum(int(a==t) for (a, _, _) in temporal_edges)
    temp_deg_per_t[t] = count_t * 2 / N      # each contact is undirected, degree to both partners
mean_deg_temporal = temp_deg_per_t.mean()
mean_deg2_temporal = (temp_deg_per_t**2).mean()

# Edge weight distribution
edge_weights = np.array([d['weight'] for _, _, d in G_static_w.edges(data=True)])

# Save diagnostics
diag_path = os.path.join(os.getcwd(), 'output', 'exp3-network-diagnostics.txt')
with open(diag_path, 'w') as f:
    f.write(f'N={N}, m={m}, T={T}, activity_gamma={gamma}, activity_min={activity_min}, activity_max={activity_max}\n')
    f.write(f'Mean activity <a>={activity.mean():.5f}, activity 2nd moment <a^2>={a2:.5f}\n')
    f.write(f'Temporal: mean deg/t={mean_deg_temporal:.5f}, 2nd moment={mean_deg2_temporal:.6f}\n')
    f.write(f'Static weighted: mean degree={mean_deg:.5f}, 2nd moment={mean_deg2:.5f}, GCC size={len(max(nx.connected_components(G_static_w), key=len))}\n')
    f.write(f'Edge weight: mean={edge_weights.mean():.5f}, median={np.median(edge_weights):.4f}, max={edge_weights.max():.2f}, min={edge_weights.min():.4f}\n')
    f.write(f'Assortativity={nx.degree_assortativity_coefficient(G_static_w):.5f}, clustering={nx.transitivity(G_static_w):.5f}\n')
    try:
        gcc_nodes = max(nx.connected_components(G_static_w), key=len)
        gcc = G_static_w.subgraph(gcc_nodes)
        f.write(f'Avg shortest path (GCC): {nx.average_shortest_path_length(gcc):.3f}\n')
    except Exception as e:
        f.write('Avg shortest path (GCC): Error '+str(e)+'\n')

# PLOTS
plt.figure()
plt.hist(activity, bins=60, color='orange', edgecolor='k', log=True)
p_act_path = os.path.join(os.getcwd(), 'output', 'exp3-activity-hist.png')
plt.xlabel('Activity a'); plt.ylabel('Count (log)'); plt.title('Power-law Activities')
plt.savefig(p_act_path, bbox_inches='tight'); plt.close()

plt.figure()
plt.hist(strengths, bins=60, color='dodgerblue', edgecolor='k', log=True)
p_strength_path = os.path.join(os.getcwd(), 'output', 'exp3-static-strength-hist.png')
plt.xlabel('Node strength (sum of edge weights)'); plt.ylabel('Count (log)')
plt.title('Static Weighted Network: Node Strength Distribution')
plt.savefig(p_strength_path, bbox_inches='tight'); plt.close()

plt.figure()
plt.hist(edge_weights, bins=60, color='crimson', edgecolor='k', log=True)
p_weight_path = os.path.join(os.getcwd(), 'output', 'exp3-static-weight-hist.png')
plt.xlabel('Edge weight'); plt.ylabel('Count (log)'); plt.title('Static Network Edge Weight Distribution')
plt.savefig(p_weight_path, bbox_inches='tight'); plt.close()

# Return outputs
diag = {
    'mean_deg': float(mean_deg),
    'mean_deg2': float(mean_deg2),
    'mean_deg_temporal': float(mean_deg_temporal),
    'mean_deg2_temporal': float(mean_deg2_temporal),
    'mean_activity': float(activity.mean()),
    'act2': float(a2),
    'number_nodes': N,
    'gcc_size': len(max(nx.connected_components(G_static_w), key=len)),
    'number_of_edges': G_static_w.number_of_edges(),
    'mean_edge_weight': float(edge_weights.mean())
}
diag, [activity_path, edge_t_path, static_network_path, diag_path, p_act_path, p_strength_path, p_weight_path]