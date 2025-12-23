
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