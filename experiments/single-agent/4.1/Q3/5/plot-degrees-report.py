
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import sparse
import os

# Load networks
G_agg = nx.Graph(sparse.load_npz(os.path.join('output','network-agg.npz')))
G_static = nx.Graph(sparse.load_npz(os.path.join('output','network-unweighted.npz')))

agg_degrees = [d for n,d in G_agg.degree()]
static_degrees = [d for n,d in G_static.degree()]

plt.figure(figsize=(8,5))
plt.hist(agg_degrees, bins=20, alpha=0.7, label='Aggregated')
plt.hist(static_degrees, bins=20, alpha=0.7, label='Unweighted')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.title('Degree Distribution: Aggregated vs Static (Unweighted)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('output','figure-degree-dist.png'))
plt.close()