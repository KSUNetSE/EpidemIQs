
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import os
from scipy import sparse

# Parameters
N = 10000
mean_k = 3
second_moment = 15
required_p10 = 0.10  # At least 10% degree-10
possible_degrees = np.array([1, 2, 3, 4, 10])

# Set up: P(10)=0.10, now solve for P(1),..,P(4)
A = np.array([
    [1, 1, 1, 1],
    [1, 2, 3, 4],
    [1, 4, 9, 16]
])
b = np.array([0.9, 2, 5])
from numpy.linalg import lstsq
sol, residuals, rank, s = lstsq(A, b, rcond=None)

pvals = np.append(sol, required_p10)
pvals = np.clip(pvals, 0, 1)  # Ensure probabilities are valid and nonnegative
pvals = pvals / pvals.sum()  # Normalize exactly

# Now use pvals to sample degrees
np.random.seed(0)
degree_choices = np.random.choice([1,2,3,4,10], size=N, p=pvals)
if degree_choices.sum() % 2 != 0:
    # Make sum of degrees even
    idx = np.random.choice(N)
    curr_d = degree_choices[idx]
    if curr_d > 1:
        degree_choices[idx] -= 1
    else:
        degree_choices[idx] += 1

deg_hist = Counter(degree_choices)
# Empirical degree statistics
mean_k_emp = degree_choices.mean()
second_moment_emp = np.mean(degree_choices**2)
p10_emp = np.sum(degree_choices==10)/N

# Construct configuration model network
G_raw = nx.configuration_model(degree_choices, create_using=None, seed=0)
# Convert to simple undirected graph (no self-loops, multiedges)
G = nx.Graph(G_raw)  # Removes parallel edges
G.remove_edges_from(nx.selfloop_edges(G))

# Recompute mean degree etc. after pruning
degree_list = [d for n,d in G.degree()]
mean_k_net = np.mean(degree_list)
second_moment_net = np.mean(np.array(degree_list)**2)
p10_net = np.mean(np.array(degree_list)==10)

# Diagnostics
GCC = max(nx.connected_components(G), key=len)
GCC_frac = len(GCC)/N
clustering = nx.average_clustering(G)
assortativity = nx.degree_assortativity_coefficient(G)

# Save network as sparse matrix
outdir = os.path.join(os.getcwd(), 'output')
os.makedirs(outdir, exist_ok=True)
net_path = os.path.join(outdir, 'network-cm-mixeddegree.npz')
sparse.save_npz(net_path, nx.to_scipy_sparse_array(G))

# Degree distribution plot
plt.figure(figsize=(6,4))
counts, bins = np.histogram(degree_list, bins=np.arange(0,max(degree_list)+2)-0.5)
plt.bar(bins[:-1], counts/len(degree_list), width=0.85, color='deepskyblue')
plt.xlabel('Degree k')
plt.ylabel('Fraction of nodes')
plt.title('Degree Distribution (Configuration Model)')
plt.tight_layout()
hist_path = os.path.join(outdir, 'degreedist-cm-mixeddegree.png')
plt.savefig(hist_path)
plt.close()

# GCC Structure plot (subsample)
plt.figure(figsize=(6,3))
GCC_subG = G.subgraph(GCC)
pos = nx.spring_layout(GCC_subG, seed=1)
visual_nodes = np.random.choice(list(GCC_subG.nodes), size=min(200,len(GCC_subG)), replace=False)
nx.draw_networkx_nodes(GCC_subG, pos, nodelist=visual_nodes, node_color='orangered', node_size=14)
nx.draw_networkx_edges(GCC_subG, pos, edgelist=[e for e in GCC_subG.edges(visual_nodes)], alpha=0.2, width=0.5)
plt.title('Largest Component (sampled nodes)')
plt.axis('off')
GCC_plot_path = os.path.join(outdir, 'GCC-cm-mixeddegree.png')
plt.tight_layout()
plt.savefig(GCC_plot_path)
plt.close()

results = {
    'net_path': net_path,
    'hist_path': hist_path,
    'gcc_path': GCC_plot_path,
    'degree_probs': dict(zip([1,2,3,4,10], pvals)),
    'mean_k_sampled': float(mean_k_emp),
    'second_moment_sampled': float(second_moment_emp),
    'p10_sampled': float(p10_emp),
    'mean_k_simple': float(mean_k_net),
    'second_moment_simple': float(second_moment_net),
    'p10_simple': float(p10_net),
    'gcc_frac': float(GCC_frac),
    'clustering': float(clustering),
    'assortativity': float(assortativity),
    'N': N
}

results
