
import numpy as np
import pandas as pd
import os
import networkx as nx
from scipy import sparse
import random
import matplotlib.pyplot as plt

N = 1000
alpha = 0.1
m = 5
T = 1000
all_edges = []
temporal_edge_events = []
edge_counts = dict()  # For static aggregation

for t in range(T):
    # Nodes that activate this timestep
    active_nodes = [i for i in range(N) if random.random() < alpha]
    for src in active_nodes:
        targets = set()
        while len(targets) < m:
            tgt = random.randint(0, N-1)
            if tgt != src:
                targets.add(tgt)
        for tgt in targets:
            a, b = sorted([src, tgt])
            temporal_edge_events.append({'time': t, 'src': a, 'tgt': b})
            all_edges.append((a,b))
            key = (a,b)
            edge_counts[key] = edge_counts.get(key,0)+1

# Save as CSV edge-event list
out_folder = os.path.join(os.getcwd(), 'output')
os.makedirs(out_folder, exist_ok=True)
temporal_event_path = os.path.join(out_folder, 'temporal-edge-events-csv')
pd.DataFrame(temporal_edge_events).to_csv(temporal_event_path, index=False)

# Plot node activity histogram (total activations per node)
node_acts = [0]*N
for e in temporal_edge_events:
    node_acts[e['src']] += 1
    node_acts[e['tgt']] += 1
activity_hist_path = os.path.join(out_folder, 'node-activity-temporal.png')
plt.figure(figsize=(5,3))
plt.hist(node_acts, bins=40, color='purple')
plt.xlabel('Total times node participated in contact')
plt.ylabel('Node count')
plt.tight_layout()
plt.savefig(activity_hist_path)
plt.close()

# Aggregate: Build weighted adjacency matrix for cumulative contacts
g_static = nx.Graph()
g_static.add_nodes_from(range(N))
edge_weights = []
for (a,b),w in edge_counts.items():
    g_static.add_edge(a,b,weight=w)
    edge_weights.append(w)

# Compute mean degree and 2nd degree moment (weighted degree)
k_arr = np.array([d for n,d in g_static.degree()])
mean_k = float(np.mean(k_arr))
mean_k2 = float(np.mean(k_arr**2))

# Plot degree distribution
degplot_path = os.path.join(out_folder,'degreedist-agg-static.png')
plt.figure(figsize=(5,3))
plt.hist(k_arr, bins=40, color='teal')
plt.xlabel('Degree (static)')
plt.ylabel('Node count')
plt.tight_layout()
plt.savefig(degplot_path)
plt.close()

# Plot weight distribution (edge weights)
weightplot_path = os.path.join(out_folder,'edgeweight-agg-static.png')
plt.figure(figsize=(5,3))
plt.hist(edge_weights, bins=40, color='orange')
plt.xlabel('Edge weight (contact freq.)')
plt.ylabel('Edge count')
plt.tight_layout()
plt.savefig(weightplot_path)
plt.close()

# Save static weighted network as adjacency matrix
agg_static_net_path = os.path.join(out_folder, 'agg-static-network-npz')
sparse.save_npz(agg_static_net_path, nx.to_scipy_sparse_array(g_static, weight='weight'))

stats = {'agg_static_net_path': agg_static_net_path,
         'mean_k': mean_k,
         'mean_k2': mean_k2,
         'degplot_path': degplot_path,
         'weightplot_path': weightplot_path,
         'temporal_event_path': temporal_event_path,
         'activity_hist_path': activity_hist_path}
