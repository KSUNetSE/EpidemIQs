
import numpy as np, networkx as nx, scipy.sparse as sparse, os, json
output_dir=os.path.join(os.getcwd(),'output')
G=sparse.load_npz(os.path.join(output_dir,'network.npz'))
G_nx=nx.from_scipy_sparse_array(G)

k_vals = np.array([d for n,d in G_nx.degree()])
mean_k = k_vals.mean()
second = (k_vals**2).mean()

deg10_idx = np.where(k_vals==10)[0]
P10 = len(deg10_idx)/len(k_vals)
print('P10',P10)

others_idx = np.where(k_vals!=10)[0]

k_others = k_vals[others_idx]

mean_other = k_others.mean()
second_other = (k_others**2).mean()

def q_after(alpha):
    # remove alpha fraction of deg10 nodes
    remaining_deg10 = (1-alpha)*10
    # Actually nodes removed entirely, so for degrees we keep only (1-alpha) fraction of deg10 nodes.
    N=len(k_vals)
    # aggregated moments
    mean_kp = ( (1-alpha)*P10*10 + (1-P10)*mean_other ) / (1 - alpha*P10)
    second_p = ( (1-alpha)*P10*10**2 + (1-P10)*second_other ) / (1 - alpha*P10)
    q_p = (second_p - mean_kp)/mean_kp
    return q_p

alphas = np.linspace(0,1,21)
qs=[q_after(a) for a in alphas]
print(list(zip(alphas,qs)))

# find alpha where q<1
for a,qx in zip(alphas,qs):
    if qx<1:
        print('need alpha',a)
        break