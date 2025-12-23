
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import random
np.random.seed(42)
random.seed(42)

# Step 1. Parameters
N = 10000
mean_k = 10
seed = 42
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

# Step 2. ER Network Construction
p = mean_k/(N-1)
g_er = nx.erdos_renyi_graph(N, p, seed=seed)

# Step 3. Scale-Free Network Construction using Configuration model
# Power-law exponent
gamma = 2.7
kmin = 2
# Generate power-law degree sequence, constrained to sum even and mean_k
max_trials = 1000
for trial in range(max_trials):
    deg_seq = np.random.zipf(gamma, N)
    deg_seq = np.clip(deg_seq, kmin, 1000)  # avoid degree 1, cap unphysically high deg
    # Adjust for correct mean degree
    factor = mean_k / np.mean(deg_seq)
    deg_seq = np.round(deg_seq * factor).astype(int)
    deg_seq = np.clip(deg_seq, kmin, N // 2)
    if sum(deg_seq) % 2 == 1:
        deg_seq[random.randrange(N)] += 1
    if np.abs(np.mean(deg_seq) - mean_k) < 0.2:
        break
else:
    raise ValueError("Could not generate degree sequence with correct mean in {} trials".format(max_trials))

Gcf = nx.configuration_model(deg_seq, seed=seed)
Gcf = nx.Graph(Gcf)  # Remove parallel edges
Gcf.remove_edges_from(nx.selfloop_edges(Gcf))

# Use largest connected component (LCC) if not fully connected
Gcf_gc = max(nx.connected_components(Gcf), key=len)
Gcf = Gcf.subgraph(Gcf_gc).copy()

# Step 4. Compute diagnostics
# For ER
kvals_er = [d for n, d in g_er.degree()]
mean_k_er = np.mean(kvals_er)
mean_k2_er = np.mean(np.power(kvals_er, 2))
# For Scale-free
kvals_sf = [d for n, d in Gcf.degree()]
mean_k_sf = np.mean(kvals_sf)
mean_k2_sf = np.mean(np.power(kvals_sf, 2))

# Step 5. Plots
fig1, ax1 = plt.subplots()
ax1.hist(kvals_er, bins=30, alpha=0.7, color='b')
ax1.set_title('Degree Distribution ER Random Graph')
ax1.set_xlabel('Degree (k)')
ax1.set_ylabel('Frequency')
degree_plot_er_path = os.path.join(output_dir, 'ER-DegreeHist.png')
fig1.savefig(degree_plot_er_path, dpi=150)
plt.close(fig1)

fig2, ax2 = plt.subplots()
ax2.hist(kvals_sf, bins=np.geomspace(2, max(kvals_sf), num=35), alpha=0.7, color='r')
ax2.set_xscale('log')
ax2.set_yscale('linear')
ax2.set_title('Degree Distribution Scale-Free (Config, γ=2.7)')
ax2.set_xlabel('Degree (k)')
ax2.set_ylabel('Frequency')
degree_plot_sf_path = os.path.join(output_dir, 'ScaleFree-DegreeHist.png')
fig2.tight_layout()
fig2.savefig(degree_plot_sf_path, dpi=150)
plt.close(fig2)

# Step 6. Save the networks
er_npz_path = os.path.join(output_dir, 'ER-Network.npz')
sf_npz_path = os.path.join(output_dir, 'ScaleFree-Network.npz')
sparse.save_npz(er_npz_path, nx.to_scipy_sparse_array(g_er))
sparse.save_npz(sf_npz_path, nx.to_scipy_sparse_array(Gcf))

# Extra: Determin relevant centralities/diagnostics
# For both: Compute size of giant component
gcc_size_er = len(max(nx.connected_components(g_er), key=len))
gcc_frac_er = gcc_size_er/N

gcc_size_sf = len(Gcf_gc)
gcc_frac_sf = gcc_size_sf/N

# For both: Degree assortativity and clustering
assort_er = nx.degree_assortativity_coefficient(g_er)
clust_er = nx.average_clustering(g_er)

assort_sf = nx.degree_assortativity_coefficient(Gcf)
clust_sf = nx.average_clustering(Gcf)

return_dict = {
    "N": N,
    "er_npz_path": er_npz_path,
    "sf_npz_path": sf_npz_path,
    "mean_k_er": mean_k_er,
    "mean_k2_er": mean_k2_er,
    "mean_k_sf": mean_k_sf,
    "mean_k2_sf": mean_k2_sf,
    "degree_plot_er_path": degree_plot_er_path,
    "degree_plot_sf_path": degree_plot_sf_path,
    "gcc_size_er": gcc_size_er,
    "gcc_frac_er": gcc_frac_er,
    "gcc_size_sf": gcc_size_sf,
    "gcc_frac_sf": gcc_frac_sf,
    "assort_er": assort_er,
    "clust_er": clust_er,
    "assort_sf": assort_sf,
    "clust_sf": clust_sf
}
