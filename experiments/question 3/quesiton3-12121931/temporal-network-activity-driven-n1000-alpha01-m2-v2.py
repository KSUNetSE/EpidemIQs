
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
N = 1000
alpha = 0.1
m = 2
T = 1000
# Prepare storage for the edge table and per-timestep statistics
timesteps = []
sources = []
targets = []
per_timestep_contacts = []
per_timestep_degree_counts = []
for t in range(T):
    step_deg = np.zeros(N, dtype=int)
    active_nodes = np.where(np.random.rand(N) < alpha)[0]
    edges_this_step = set()
    for u in active_nodes:
        partners = set()
        while len(partners) < m:
            v = np.random.randint(0, N)
            if v != u:
                partners.add(v)
        for v in partners:
            edge = tuple(sorted((u, v)))
            if edge not in edges_this_step:
                edges_this_step.add(edge)
                timesteps.append(t)
                sources.append(edge[0])
                targets.append(edge[1])
                step_deg[edge[0]] += 1
                step_deg[edge[1]] += 1
    per_timestep_contacts.append(len(edges_this_step))
    per_timestep_degree_counts.append(step_deg)
# Save edge table
df_edges = pd.DataFrame({'timestep': timesteps, 'source': sources, 'target': targets})
output_dir = os.path.join(os.getcwd(), 'output')
temporal_network_path = os.path.join(output_dir, 'temporal-edge-table-n1000-alpha01-m2-T1000.csv')
df_edges.to_csv(temporal_network_path, index=False)
# Mean degree and <k^2>
all_degrees = np.stack(per_timestep_degree_counts)
mean_degree = all_degrees.mean()
k2 = np.mean(all_degrees**2)
# Plots
deg_flat = all_degrees.flatten()
plt.figure(figsize=(6,4))
plt.hist(deg_flat, bins=np.arange(deg_flat.max()+2)-0.5, alpha=0.7)
plt.xlabel('Node Degree in One Timestep')
plt.ylabel('Count')
plt.title('Degree Distribution Per Timestep (Temporal)')
plt.savefig(os.path.join(output_dir, 'temporal-degree-histogram.png'))
plt.close()
plt.figure(figsize=(6,4))
plt.plot(per_timestep_contacts)
plt.xlabel('Timestep')
plt.ylabel('Number of unique edges')
plt.title('Number of Edges per Timestep (Temporal)')
plt.savefig(os.path.join(output_dir, 'temporal-contacts-tseries.png'))
plt.close()
return_vars = [
    'temporal_network_path',
    'mean_degree',
    'k2',
    'output_dir'
]