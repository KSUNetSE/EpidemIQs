
import numpy as np
import random
import networkx as nx
import pandas as pd
import os
from collections import defaultdict
from scipy import sparse
import matplotlib.pyplot as plt

# Parameters
dir_out = os.path.join(os.getcwd(), "output")
N = 1000
T = 100
alpha = 0.1
m = 2

random.seed(42)
np.random.seed(42)

# For temporal network: record edge list per time step
temporal_edges = [] # (t, i, j) tuples
node_activation_count = np.zeros(N, dtype=int)
contacts_per_node = np.zeros(N, dtype=int)
contacts_per_time = []

for t in range(T):
    active_nodes = [i for i in range(N) if random.random() < alpha]
    chosen_edges = set()
    for i in active_nodes:
        node_activation_count[i] += 1
        partners = set()
        while len(partners) < m:
            j = random.randint(0, N-1)
            if j != i:
                if (i,j) not in chosen_edges and (j,i) not in chosen_edges and j not in partners:
                    partners.add(j)
                    temporal_edges.append( (t, i, j) )
                    chosen_edges.add((i,j))
        contacts_per_node[i] += m
        for j in partners:
            contacts_per_node[j] += 1
    contacts_per_time.append(len(chosen_edges))

temporal_df = pd.DataFrame(temporal_edges, columns=['time','src','tgt'])
temporal_edge_path = os.path.join(dir_out, "temporal-edgetable.csv")
temporal_df.to_csv(temporal_edge_path, index=False)

# For aggregate (weighted) network
agg_matrix = np.zeros((N, N), dtype=int)
for _, i, j in temporal_edges:
    agg_matrix[i, j] += 1
    agg_matrix[j, i] += 1 # undirected
G_agg = nx.from_numpy_array(agg_matrix)
agg_weighted_path = os.path.join(dir_out, "aggregated-network.npz")
sparse.save_npz(agg_weighted_path, sparse.csr_matrix(agg_matrix))

# Aggregated network degree (count positive unique partners):
degrees = [ np.count_nonzero(agg_matrix[i,:]) for i in range(N) ]
mean_deg = np.mean(degrees)
second_deg_moment = np.mean(np.array(degrees)**2)

# For weighted degree distribution (sum weight of edges per node)
weighted_degs = np.sum(agg_matrix, axis=1)

# Diagnostics & visualizations
plt.figure(figsize=(6,4))
plt.hist(weighted_degs, bins=35, alpha=0.7)
plt.xlabel('Weighted Degree (Total Contacts)')
plt.ylabel('Number of Nodes')
plt.title('Distribution of Weighted Degrees (Aggregated Network)')
plt.tight_layout()
agg_weighted_plot_path = os.path.join(dir_out, "weighted-degree-distribution.png")
plt.savefig(agg_weighted_plot_path)
plt.close()

plt.figure(figsize=(6,4))
plt.hist(degrees, bins=30, alpha=0.7)
plt.xlabel('Unique Partners (Aggregated Degree)')
plt.ylabel('Nodes')
plt.title('Distribution of Number of Unique Partners (Aggregated)')
plt.tight_layout()
agg_deg_plot_path = os.path.join(dir_out, "aggregated-degree-distribution.png")
plt.savefig(agg_deg_plot_path)
plt.close()

plt.figure(figsize=(6,4))
plt.hist(node_activation_count, bins=20, alpha=0.7)
plt.xlabel('Activations per Node')
plt.ylabel('Nodes')
plt.title('Distribution of Node Activations (over T)')
plt.tight_layout()
activation_plot_path = os.path.join(dir_out, "node-activation-distribution.png")
plt.savefig(activation_plot_path)
plt.close()

plt.figure(figsize=(6,4))
plt.plot(range(T), contacts_per_time, marker='.')
plt.xlabel('Time step')
plt.ylabel('Contacts per time step')
plt.title('Temporal Contacts Per Time step')
plt.tight_layout()
time_contact_plot_path = os.path.join(dir_out, "contacts-per-timestep.png")
plt.savefig(time_contact_plot_path)
plt.close()

# Compute temporal moments (per time step)
time_mean_degrees = [contacts / N for contacts in contacts_per_time] # at t, mean degree per node (times 2)
time_second_degrees = [ (contacts*2) / N for contacts in contacts_per_time] # as approximation
avg_dynamic_deg = np.mean(time_mean_degrees)
avg_dynamic_deg2 = np.mean(np.square(time_mean_degrees))

results = {
    'temporal_edge_path': temporal_edge_path,
    'agg_weighted_path': agg_weighted_path,
    'agg_weighted_plot_path': agg_weighted_plot_path,
    'agg_deg_plot_path': agg_deg_plot_path,
    'activation_plot_path': activation_plot_path,
    'time_contact_plot_path': time_contact_plot_path,
    'mean_deg': mean_deg,
    'second_deg_moment': second_deg_moment,
    'avg_dynamic_deg': avg_dynamic_deg,
    'avg_dynamic_deg2': avg_dynamic_deg2
}

results
