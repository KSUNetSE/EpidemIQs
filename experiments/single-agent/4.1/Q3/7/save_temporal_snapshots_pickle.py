
import numpy as np
import pickle
import os

# Parameters
N = 1000
alpha = 0.1
m = 2
T = 500

def make_temporal_snapshots(N, alpha, m, T):
    edge_snapshots = []
    for t in range(T):
        active = np.random.rand(N) < alpha
        active_nodes = np.where(active)[0]
        new_edges = []
        for src in active_nodes:
            targets = np.random.choice(np.delete(np.arange(N), src), m, replace=False)
            for dst in targets:
                edge = tuple(sorted([src, dst]))
                new_edges.append(edge)
        edge_snapshots.append(new_edges)
    return edge_snapshots

snapshots = make_temporal_snapshots(N, alpha, m, T)
with open(os.path.join(os.getcwd(), 'output', 'temporal_snapshots.pkl'), 'wb') as f:
    pickle.dump(snapshots, f)