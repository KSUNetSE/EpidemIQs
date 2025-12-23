
import numpy as np, networkx as nx, scipy.sparse as sparse, os
out=os.path.join(os.getcwd(),'output')
G=sparse.load_npz(os.path.join(out,'network.npz'))
Gnx=nx.from_scipy_sparse_array(G)
ks=np.array([d for _,d in Gnx.degree()])
N=len(ks)
P10 = np.mean(ks==10)
k_other=ks[ks!=10]
mu_o=k_other.mean()
sec_o=(k_other**2).mean()
q0=(ks**2).mean()/ks.mean() -1
beta_over_gamma=4/q0

def q_after(alpha):
    denom=1 - alpha*P10
    mean_kp = ((1-alpha)*P10*10 + (1-P10)*mu_o)/denom
    second_p = ((1-alpha)*P10*100 + (1-P10)*sec_o)/denom
    return (second_p - mean_kp)/mean_kp

for alpha in np.linspace(0,1,201):
    Rp=beta_over_gamma*q_after(alpha)
    if Rp<1:
        print(alpha)
        break