
import numpy as np, networkx as nx, scipy.sparse as sparse, os, json
out=os.path.join(os.getcwd(),'output')
G=sparse.load_npz(os.path.join(out,'network.npz'))
Gnx=nx.from_scipy_sparse_array(G)
ks=np.array([d for _,d in Gnx.degree()])
N=len(ks)
p= np.mean(ks==10)
k_other = ks[ks!=10]
mu_other = k_other.mean()
sec_other = (k_other**2).mean()
q0 = (ks**2).mean()/ks.mean() -1
print('q0',q0)

beta_over_gamma = 4/q0

def q_after(alpha):
    denom = 1 - alpha*p
    mean_kp = ((1-alpha)*p*10 + (1-p)*mu_other)/denom
    second_p = ((1-alpha)*p*100 + (1-p)*sec_other)/denom
    q_p = (second_p - mean_kp)/mean_kp
    return q_p

a = np.linspace(0,1,201)
for alpha in a:
    q_p=q_after(alpha)
    R=beta_over_gamma*q_p
    if R<1:
        print('alpha_stop',alpha)
        break