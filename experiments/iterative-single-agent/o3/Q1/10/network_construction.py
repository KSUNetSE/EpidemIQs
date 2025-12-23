
import networkx as nx, numpy as np, os, scipy.sparse as sparse
N=5000
k_avg=10
p=k_avg/(N-1)
G_er=nx.fast_gnp_random_graph(N,p,seed=42)
mean_er=np.mean([d for _,d in G_er.degree()])
second_er=np.mean([d**2 for _,d in G_er.degree()])
q_er=(second_er-mean_er)/mean_er
m=k_avg//2
G_ba=nx.barabasi_albert_graph(N,m,seed=42)
mean_ba=np.mean([d for _,d in G_ba.degree()])
second_ba=np.mean([d**2 for _,d in G_ba.degree()])
q_ba=(second_ba-mean_ba)/mean_ba
output_dir=os.path.join(os.getcwd(),'output')
os.makedirs(output_dir,exist_ok=True)
sparse.save_npz(os.path.join(output_dir,'network_er.npz'),nx.to_scipy_sparse_array(G_er))
sparse.save_npz(os.path.join(output_dir,'network_ba.npz'),nx.to_scipy_sparse_array(G_ba))
print({'mean_er':mean_er,'second_er':second_er,'q_er':q_er,'mean_ba':mean_ba,'second_ba':second_ba,'q_ba':q_ba})
import networkx as nx, scipy.sparse as sparse, os, numpy as np, json, math, random, matplotlib
from pathlib import Path
current_dir=os.getcwd()
output_dir=os.path.join(current_dir,'output')
Path(output_dir).mkdir(exist_ok=True)
N=2000
# ER network with mean degree 10 -> p = k/(N-1)
mean_k=10
p=mean_k/(N-1)
G_er=nx.fast_gnp_random_graph(N,p,seed=42)
# BA network with m=5 ( gives mean degree ~2m=10)
G_ba=nx.barabasi_albert_graph(N,5,seed=42)
# compute degree moments function
import numpy as np

def degree_moments(G):
    deg=np.array([d for n,d in G.degree()])
    mean_k=deg.mean()
    k2=(deg**2).mean()
    return mean_k,k2

mean_k_er,k2_er=degree_moments(G_er)
mean_k_ba,k2_ba=degree_moments(G_ba)
# save networks
sparse.save_npz(os.path.join(output_dir,'er_network.npz'), nx.to_scipy_sparse_array(G_er))
sparse.save_npz(os.path.join(output_dir,'ba_network.npz'), nx.to_scipy_sparse_array(G_ba))
# return metrics
result={
    'mean_k_er':mean_k_er,
    'k2_er':k2_er,
    'mean_k_ba':mean_k_ba,
    'k2_ba':k2_ba,
    'er_path':os.path.join(output_dir,'er_network.npz'),
    'ba_path':os.path.join(output_dir,'ba_network.npz')
}
result
import os, networkx as nx, numpy as np, scipy.sparse as sparse
os.makedirs(os.path.join(os.getcwd(), 'output'), exist_ok=True)
# Parameters
N = 5000
mean_degree = 10
# ER network p such that expected degree = mean_degree
p = mean_degree/ (N-1)
G_er = nx.erdos_renyi_graph(N, p, seed=42)
# BA network with m edges per new node such that average degree ~2m ~ mean_degree
m = mean_degree//2  # 5
G_ba = nx.barabasi_albert_graph(N, m, seed=42)
# Save networks
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network_er.npz'), nx.to_scipy_sparse_array(G_er))
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'network_ba.npz'), nx.to_scipy_sparse_array(G_ba))
# Compute degree moments
k_er = np.array([d for n,d in G_er.degree()])
k_ba = np.array([d for n,d in G_ba.degree()])
mean_er = k_er.mean(); mean2_er = (k_er**2).mean()
mean_ba = k_ba.mean(); mean2_ba = (k_ba**2).mean()
print(mean_er, mean2_er, mean_ba, mean2_ba)

import networkx as nx
import scipy.sparse as sparse
import numpy as np, os
from pathlib import Path

# Create output directory
output_dir = Path(os.getcwd())/"output"
output_dir.mkdir(exist_ok=True)

N = 5000
mean_k = 10
p = mean_k/(N-1)
# Erdős-Rényi network
G_er = nx.fast_gnp_random_graph(N, p, seed=42, directed=False)
# Barabási-Albert network with m=5 (mean degree ~2m =10)
G_ba = nx.barabasi_albert_graph(N, 5, seed=42)

er_csr = nx.to_scipy_sparse_array(G_er, format='csr')
ba_csr = nx.to_scipy_sparse_array(G_ba, format='csr')

sparse.save_npz(output_dir/"network_er.npz", er_csr)
sparse.save_npz(output_dir/"network_ba.npz", ba_csr)

# Compute degree statistics
k_er = np.array([d for _, d in G_er.degree()])
mean_k_er = k_er.mean()
second_moment_er = np.mean(k_er**2)

k_ba = np.array([d for _, d in G_ba.degree()])
mean_k_ba = k_ba.mean()
second_moment_ba = np.mean(k_ba**2)

return_dict = {
    'mean_k_er': mean_k_er,
    'second_moment_er': second_moment_er,
    'mean_k_ba': mean_k_ba,
    'second_moment_ba': second_moment_ba,
    'n_nodes': N,
}
