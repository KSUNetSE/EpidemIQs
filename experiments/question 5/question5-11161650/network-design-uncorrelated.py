
import numpy as np
import networkx as nx
from scipy import sparse
import matplotlib.pyplot as plt
import os

# Helper function to solve for degree distribution
# Given degree support 2, 3, 10 and target mean degree and q
# For p10 = 0.12
p10 = 0.12
x = 7 * p10     # fraction with degree 2
y = 1 - 8 * p10 # fraction with degree 3
N = 10000

# Number of nodes in each degree class
n2 = int(round(x * N))
n3 = int(round(y * N))
n10 = int(round(p10 * N))
# To make sure total is N
while n2 + n3 + n10 != N:
    n3 += (N - (n2 + n3 + n10))

# Compose degree sequence
seq = [2] * n2 + [3] * n3 + [10] * n10
np.random.shuffle(seq)
# If total stubs is odd, flip last deg-3 to 2
if sum(seq) % 2 == 1:
    for i in range(len(seq)):
        if seq[i] == 3:
            seq[i] = 2
            break

# Configuration-model
G = nx.configuration_model(seq, seed=123)
G = nx.Graph(G)
G.remove_edges_from(nx.selfloop_edges(G))

# Save network
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
netpath = os.path.join(output_dir, "network.npz")
sparse.save_npz(netpath, nx.to_scipy_sparse_array(G))

# Diagnostics: degree, mean degree, second moment, q
deg = np.array([d for n, d in G.degree()])
mean_k = np.mean(deg)
mean_kkm1 = np.mean(deg * (deg - 1))
mean_excess = mean_kkm1 / mean_k
mean_k2 = np.mean(deg ** 2)

# Plot degree distribution
plt.figure()
plt.hist(deg, bins=range(1,12), rwidth=0.8, align='left', color='skyblue', edgecolor='k')
plt.xlabel('Degree')
plt.ylabel('Node fraction')
plt.title('Degree Distribution (N=10000, p10=0.12)')
plt.xticks([2,3,10])
degplot=os.path.join(output_dir, "degree-distribution.png")
plt.savefig(degplot, dpi=150)
plt.close()

# Report statistics and paths
result = dict(
    network_path=netpath,
    mean_degree=mean_k,
    second_degree_moment=mean_k2,
    mean_excess_degree=mean_excess,
    plot_path=degplot,
    fraction_deg2=np.sum(deg==2)/N,
    fraction_deg3=np.sum(deg==3)/N,
    fraction_deg10=np.sum(deg==10)/N
)
result