
import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

N = 1000  # Number of nodes
alpha = 0.1
m = 2
T_aggr = 5  # days (for aggregation)
np.random.seed(42)

# Step 1: Build activity-driven temporal network edge list for 5 days
edges = []
degr_per_step = []
for t in range(T_aggr):
    active = np.random.rand(N) < alpha
    edge_set = set()
    for i in np.where(active)[0]:
        partners = np.random.choice([n for n in range(N) if n != i], m, replace=False)
        for j in partners:
            edge = tuple(sorted((i, j)))
            if edge not in edge_set:
                edges.append((t, edge[0], edge[1]))
                edge_set.add(edge)
    degr_per_step.append(2*len(edge_set)/N)  # average degree that day

temporal_df = pd.DataFrame(edges, columns=['t', 'i', 'j'])
outdir = os.path.join(os.getcwd(), 'output')
os.makedirs(outdir, exist_ok=True)
temporal_csv = os.path.join(outdir, 'activitydriven-temporal-edgelist.csv')
temporal_df.to_csv(temporal_csv, index=False)

# Step 2: Time-aggregated (TA) static network from union of all contacts in T days
aggr_G = nx.Graph()
aggr_G.add_nodes_from(range(N))
edge_weight = defaultdict(int)
for _, i, j in edges:
    k = tuple(sorted((i, j)))
    edge_weight[k] += 1
for (i, j), w in edge_weight.items():
    aggr_G.add_edge(i, j, weight=w)

from scipy import sparse
adj_aggr = nx.to_scipy_sparse_array(aggr_G)
taggr_adj_path = os.path.join(outdir, 'ta-from-adt-aggr-adjacency.npz')
sparse.save_npz(taggr_adj_path, adj_aggr)

aggr_deg = np.array([aggr_G.degree(i) for i in aggr_G.nodes])
aggr_meandeg = aggr_deg.mean()
aggr_deg2moment = np.mean(aggr_deg ** 2)

# Step 3: Synthetic ER static network with mean degree~2 (π=2/(N-1))
pi = 2/(N-1)
ER_G = nx.erdos_renyi_graph(N, pi, seed=1234)
er_adj = nx.to_scipy_sparse_array(ER_G)
er_adj_path = os.path.join(outdir, 'ta-er-statich-graph-adjacency.npz')
sparse.save_npz(er_adj_path, er_adj)

er_deg = np.array([ER_G.degree(i) for i in ER_G.nodes])
er_meandeg = er_deg.mean()
er_deg2moment = np.mean(er_deg**2)

# Plot degree distributions
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(aggr_deg, bins=range(min(aggr_deg), max(aggr_deg)+2), alpha=0.8, density=True)
axs[0].set_title('TA (Agg) Degree Dist')
axs[1].hist(er_deg, bins=range(min(er_deg), max(er_deg)+2), alpha=0.8, density=True)
axs[1].set_title('ER Degree Dist')
plot_path = os.path.join(outdir, 'ta-degree-dists.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.close(fig)

outputs = {
    'temporal_edgelist_csv': temporal_csv,
    'ta_from_adt_adj': taggr_adj_path,
    'er_adj': er_adj_path,
    'plot_path': plot_path,
    'aggr_meandeg': aggr_meandeg,
    'aggr_deg2moment': aggr_deg2moment,
    'er_meandeg': er_meandeg,
    'er_deg2moment': er_deg2moment
}
outputs