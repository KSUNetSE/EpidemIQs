
import numpy as np
from scipy import sparse
import networkx as nx
import os
import matplotlib.pyplot as plt

# Step 3b: Tailored degree distribution to match z=3, q=4, with nonzero fraction of degree-10 for demo
N = 10000
# set up p2, p3, p10 s.t. mean=3, mean2=15, sum=1
# Let pk for k=2,3,10: m=p2, n=p3, t=p10
# m*2 + n*3 + t*10 = 3
# m*4 + n*9 + t*100 = 15
# m + n + t = 1
# Solve the system
A = np.array([[2,3,10],[4,9,100],[1,1,1]])
b = np.array([3,15,1])
sol = np.linalg.solve(A, b)
p2, p3, p10 = sol
assert np.all([p2>=0, p3>=0, p10>=0]), 'Negative probability!'

# Compose the degree sequence with the solved probabilities
nums = np.round(np.array([p2,p3,p10])*N).astype(int)
nums[-1] += (N - nums.sum())  # to ensure sum is N

deg_seq = np.concatenate([
    np.full(nums[0], 2),
    np.full(nums[1], 3),
    np.full(nums[2], 10)
])
np.random.shuffle(deg_seq)
if deg_seq.sum()%2==1:    # Ensure even sum
    deg_seq[-1] += 1

# Build network
G_ta = nx.configuration_model(deg_seq, seed=42)
G_ta = nx.Graph(G_ta)
G_ta.remove_edges_from(nx.selfloop_edges(G_ta))

# Diagnostics
actual_degrees_ta = [d for n,d in G_ta.degree()]
mean_k_ta = np.mean(actual_degrees_ta)
mean_k2_ta = np.mean(np.array(actual_degrees_ta)**2)
q_ta = (mean_k2_ta-mean_k_ta)/mean_k_ta
num_deg10_ta = sum([1 for k in actual_degrees_ta if k==10])
fraction_deg10_ta = num_deg10_ta/N
lcc_ta = max(nx.connected_components(G_ta),key=len)
size_lcc_ta = len(lcc_ta)
assort_ta = nx.degree_assortativity_coefficient(G_ta)
clus_ta = nx.average_clustering(G_ta)

output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir,exist_ok=True)
file_ta = os.path.join(output_dir,"tailored-z3-q4-network.npz")
sparse.save_npz(file_ta, nx.to_scipy_sparse_array(G_ta))

plt.figure(figsize=(8,6))
plt.hist(actual_degrees_ta, bins=range(2,12),  density=True,color='salmon',edgecolor='k',alpha=0.7)
plt.xlabel("Degree k")
plt.ylabel("P(k)")
plt.title("Degree Distribution (Tailored: 2,3,10, z=3, q=4, N=10,000)")
degdist_ta = os.path.join(output_dir,"degree-dist-tailored-z3-q4.png")
plt.savefig(degdist_ta)
plt.close()

metrics_ta = {
    'mean_degree': mean_k_ta,
    'second_moment': mean_k2_ta,
    'mean_excess_degree': q_ta,
    'num_deg10': num_deg10_ta,
    'fraction_deg10': fraction_deg10_ta,
    'size_lcc': size_lcc_ta,
    'assortativity': assort_ta,
    'clustering': clus_ta,
    'network_file': file_ta,
    'degree_dist_plot': degdist_ta
}

metrics_ta