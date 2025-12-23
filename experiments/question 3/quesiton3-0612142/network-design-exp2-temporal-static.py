
# Now proceed to build both temporal and static networks with this correctly sampled activity array
import os
import random
import networkx as nx
from scipy import sparse
import matplotlib.pyplot as plt

random.seed(43)
np.random.seed(43)

N = 10000
T = 1000
m = 1
activity = np.load(os.path.join(os.getcwd(), 'output', 'exp2-activity-data.npy')) if os.path.exists(os.path.join(os.getcwd(), 'output', 'exp2-activity-data.npy')) else None
if activity is None:
    # fallback (from prior code block)
    r = np.random.rand(N)
    gamma = 2.5
    exp = 1-gamma
    activity_min = 0.005
    activity_max = 1.0
    amin_exp = activity_min ** exp
    amax_exp = activity_max ** exp
    activity = (r * (amax_exp - amin_exp) + amin_exp) ** (1 / exp)
    activity = np.clip(activity, activity_min, activity_max)
    np.save(os.path.join(os.getcwd(), 'output', 'exp2-activity-data.npy'), activity)

activity_path = os.path.join(os.getcwd(), 'output', 'exp2-activity-data.npy')

# TEMPORAL CONTACT SEQUENCE: Store edges (t, i, j) for each contact
edge_t_path = os.path.join(os.getcwd(), 'output', 'exp2-temporal-contact-edges.csv')
if not os.path.exists(edge_t_path):
    temporal_edges = []
    for t in range(T):
        active_nodes = [i for i in range(N) if np.random.rand() < activity[i]]
        for i in active_nodes:
            partners = set()
            while len(partners) < m:
                partner = random.randint(0, N-1)
                if partner != i:
                    partners.add(partner)
            for j in partners:
                edge = (t, min(i, j), max(i, j))
                temporal_edges.append(edge)
    with open(edge_t_path, 'w') as f:
        f.write('t,i,j\n')
        for t, i, j in temporal_edges:
            f.write(f'{t},{i},{j}\n')
else:
    temporal_edges = []
    with open(edge_t_path, 'r') as f:
        next(f)
        for line in f:
            t, i, j = map(int, line.strip().split(','))
            temporal_edges.append((t, i, j))

# Aggregated Static Random Network from Poisson degree sequence (deg_i = Poisson(mTa_i))
deg_seq = np.random.poisson(m * T * activity)
stublist = []
for idx, k in enumerate(deg_seq):
    stublist.extend([idx] * k)
random.shuffle(stublist)
static_edges = set()
for s in range(0, len(stublist) - 1, 2):
    u, v = stublist[s], stublist[s + 1]
    if u != v:
        edge = (min(u, v), max(u, v))
        static_edges.add(edge)
G_static = nx.Graph()
G_static.add_nodes_from(range(N))
G_static.add_edges_from(static_edges)
static_network_path = os.path.join(os.getcwd(), 'output', 'exp2-static-random.npz')
sparse.save_npz(static_network_path, nx.to_scipy_sparse_array(G_static))

# Diagnostics
avg_deg = np.mean([G_static.degree(n) for n in G_static.nodes])
avg_deg2 = np.mean([(G_static.degree(n)) ** 2 for n in G_static.nodes])
activity_mean = np.mean(activity)
activity_sq_mean = np.mean(activity ** 2)

# Temporal degree: degree per snapshot
temporal_degree_per_t = []
for t in range(T):
    # count unique nodes in contacts to get avg degree per snapshot
    edges_t = [e for e in temporal_edges if e[0] == t]
    temporal_degree_per_t.append(2 * len(edges_t) / N)
mean_deg_temporal = np.mean(temporal_degree_per_t)
mean_deg2_temporal = np.mean(np.array(temporal_degree_per_t) ** 2)

# Save diagnostic summary
diag_path = os.path.join(os.getcwd(), 'output', 'exp2-network-diagnostics.txt')
with open(diag_path, 'w') as f:
    f.write(f'Power-law activity: gamma=2.5, min=0.005, max=1\n')
    f.write(f'Activity mean: {activity_mean:.5f}, Activity sq. mean: {activity_sq_mean:.6f}\n')
    f.write(f'Static network: mean deg: {avg_deg:.3f}, deg2: {avg_deg2:.3f}, GCC size: {len(max(nx.connected_components(G_static), key=len))}\n')
    f.write(f'Temporal network: mean deg/timestep: {mean_deg_temporal:.5f}, deg2 moment (over t): {mean_deg2_temporal:.6f}\n')
    f.write(f'Num nodes: {N}, Num stat edges: {G_static.number_of_edges()}\n')
    f.write(f'Assortativity (deg): {nx.degree_assortativity_coefficient(G_static):.5f}\n')
    try:
        gcc_nodes = max(nx.connected_components(G_static), key=len)
        gcc = G_static.subgraph(gcc_nodes)
        f.write(f'Avg shortest path (GCC): {nx.average_shortest_path_length(gcc):.3f}\n')
    except Exception as e:
        f.write('Avg shortest path (GCC): Error ' + str(e) + '\n')
    f.write(f'Global clustering: {nx.transitivity(G_static):.5f}\n')

# Activity histogram
plt.figure()
plt.hist(activity, bins=60, color='cornflowerblue', edgecolor='k', log=True)
p_act_path = os.path.join(os.getcwd(), 'output', 'exp2-activity-hist.png')
plt.xlabel('Activity a'); plt.ylabel('Count (log)'); plt.title('Power-law Activity Rates')
plt.savefig(p_act_path, bbox_inches='tight'); plt.close()

# Degree histogram for static
plt.figure()
deg_list = [G_static.degree(n) for n in G_static.nodes]
plt.hist(deg_list, bins=60, color='orchid', edgecolor='k', log=True)
p_deg_path = os.path.join(os.getcwd(), 'output', 'exp2-static-deg-hist.png')
plt.xlabel('Degree'); plt.ylabel('Count (log)'); plt.title('Degree Distribution (Static Aggregated Random)')
plt.savefig(p_deg_path, bbox_inches='tight'); plt.close()

diag = {
    'mean_deg': avg_deg,
    'mean_deg2': avg_deg2,
    'mean_deg_temporal': mean_deg_temporal,
    'mean_deg2_temporal': mean_deg2_temporal,
    'mean_activity': activity_mean,
    'act2': activity_sq_mean,
    'number_nodes': N,
    'gcc_size': len(max(nx.connected_components(G_static), key=len)),
    'number_of_edges': G_static.number_of_edges(),
}

diag, [activity_path, edge_t_path, static_network_path, diag_path, p_act_path, p_deg_path]
