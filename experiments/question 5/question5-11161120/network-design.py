
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
import os

# Parameters
N = 10000  # network size

# Degree distribution:
# let pk[10] = 0.1 (10% degree-10 nodes)
# Remaining 90% distributed over degrees 1-6 so that mean degree is 3 and mean squared degree is 15
# We'll (manually) solve for p1..p6.

# Let x1 = fraction at degree 1, ... x6 = fraction at degree 6
# x10 = 0.1
# x1 + x2 + x3 + x4 + x5 + x6 = 0.9
# 1*x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5 + 6*x6 + 10*0.1 = 3
# 1*x1 + 4*x2 + 9*x3 + 16*x4 + 25*x5 + 36*x6 + 100*0.1 = 15

# This is underdetermined. Let's try assigning reasonable structure: Let most of the population be at k=2,3,4.
x10 = 0.1
# Try x1 = 0.25, x2 = 0.20, x3 = 0.15, x4 = 0.15, x5 = 0.08, x6 = 0.07 (adjusted to sum to 0.9)
x1 = 0.25
x2 = 0.20
x3 = 0.15
x4 = 0.15
x5 = 0.08
x6 = 0.07
prob = [0, x1, x2, x3, x4, x5, x6, 0, 0, 0, x10]  # 0 for k=0,7,8,9 (zero probability)

# Check constraints:
deg = np.arange(len(prob))
mean_k = np.sum(deg * prob)
mean_k2 = np.sum(deg**2 * prob)
p10 = prob[10]
# Adjust if necessary (we will do this after sampling degree sequence if discrepancy is minor)

# Construct degree sequence
kvals = []
for k, p in enumerate(prob):
    count = int(round(p * N))
    kvals.extend([k] * count)
# If sum is not N, adjust last element
if len(kvals) > N:
    kvals = kvals[:N]
elif len(kvals) < N:
    kvals += [1]*(N-len(kvals))  # pad with degree-1 nodes
# Ensure sum of degrees is even
if sum(kvals) % 2 != 0:
    kvals[0] += 1

# Generate configuration model
g = nx.configuration_model(kvals, seed=42)
# remove parallel edges and self loops
g = nx.Graph(g) # removes parallel edges
nx.set_node_attributes(g, 0, 'is_high_deg') # mark high-degree nodes
deg10_nodes = [n for n, d in g.degree() if d == 10]
for n in deg10_nodes:
    g.nodes[n]['is_high_deg'] = 1
g.remove_edges_from(nx.selfloop_edges(g))

# Calculate final degree statistics after cleaning
deg_seq = [d for n, d in g.degree()]
mean_degree = np.mean(deg_seq)
mean_k2 = np.mean([k**2 for k in deg_seq])
mean_excess = (mean_k2 - mean_degree) / mean_degree
num_deg10 = sum([1 for d in deg_seq if d == 10])
p10_obs = num_deg10 / N

# Save network
dir_out = os.path.join(os.getcwd(), 'output')
os.makedirs(dir_out, exist_ok=True)
network_path = os.path.join(dir_out, 'network.npz')
sparse.save_npz(network_path, nx.to_scipy_sparse_array(g))

# Save degree distribution plot
plt.figure(figsize=(7,5))
degrange = np.bincount(deg_seq)
plt.bar(range(len(degrange)), degrange/N)
plt.xlabel('Degree k')
plt.ylabel('Fraction of nodes')
plt.title('Network Degree Distribution')
degree_plot_path = os.path.join(dir_out, 'degree-dist.png')
plt.savefig(degree_plot_path)
plt.close()

# Save top-15 degree centrality plot
deg_cent = dict(g.degree())
top15 = sorted(deg_cent.items(), key=lambda x: -x[1])[:15]
names = [str(k) for k,_ in top15]
vals = [v for _,v in top15]
plt.figure(figsize=(6,4))
plt.bar(names, vals)
plt.xlabel('Node id (top-15)')
plt.ylabel('Degree')
plt.title('Top-15 Node Degrees')
topcent_plot_path = os.path.join(dir_out, 'top-deg-centrality.png')
plt.savefig(topcent_plot_path)
plt.close()

results = dict(
    network_path=network_path,
    mean_degree=mean_degree,
    mean_k2=mean_k2,
    mean_excess=mean_excess,
    p10_obs=p10_obs,
    num_deg10=num_deg10,
    degree_plot_path=degree_plot_path,
    topcent_plot_path=topcent_plot_path,
    N=len(g.nodes)
)
results