
import numpy as np, networkx as nx, scipy.sparse as sparse, os
output_dir=os.path.join(os.getcwd(),'output')
G=sparse.load_npz(os.path.join(output_dir,'network.npz'))
G_nx=nx.from_scipy_sparse_array(G)

k_vals=np.array([d for n,d in G_nx.degree()])
N=len(k_vals)
P10=np.mean(k_vals==10)

k_other = k_vals[k_vals!=10]
mean_other = k_other.mean()
second_other = (k_other**2).mean()

beta_over_gamma=4/((k_vals**2).mean()/k_vals.mean() -1)
print('beta_over_gamma',beta_over_gamma)


def q_after(alpha):
    # remove alpha fraction of deg10 nodes
    remain_factor = 1 - alpha
    # remaining nodes fraction denom= 1 - alpha*P10
    denom = 1 - alpha*P10
    mean_kp = (remain_factor*P10*10 + (1-P10)*mean_other)/denom
    second_p = (remain_factor*P10*100 + (1-P10)*second_other)/denom
    q_p = (second_p - mean_kp)/mean_kp
    return q_p

a= np.linspace(0,1,101)
qs=[q_after(alpha) for alpha in a]
Rpr=[beta_over_gamma*q for q in qs]
thr=[(beta_over_gamma*q) for q in qs]
for alpha,q,R in zip(a,qs,Rpr):
    if R<1:
        print('alpha needed',alpha, 'q',q,'R',R)
        break