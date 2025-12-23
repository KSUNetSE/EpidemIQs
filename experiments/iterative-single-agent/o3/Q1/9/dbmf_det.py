
import numpy as np, os, collections, networkx as nx, scipy.sparse as sparse
output_dir = os.path.join(os.getcwd(),'output')
G_er = nx.from_scipy_sparse_array(sparse.load_npz(os.path.join(output_dir,'er_network.npz')))
G_ba = nx.from_scipy_sparse_array(sparse.load_npz(os.path.join(output_dir,'ba_network.npz')))

sigma = 1/5
gamma = 1/7
beta_er = 0.04479449488691632
beta_ba = 0.019742496911525718
N=2000
I0=10

def degree_distribution(G):
    deg = [d for _,d in G.degree()]
    counts = collections.Counter(deg)
    Pk={k: c/len(deg) for k,c in counts.items()}
    return Pk
Pk_er = degree_distribution(G_er)
Pk_ba = degree_distribution(G_ba)


def integrate_dbmf(Pk,beta,T=180,dt=0.1):
    ks = sorted(Pk.keys())
    num_classes = len(ks)
    # initial fractions per degree class
    S = np.ones(num_classes)
    E = np.zeros(num_classes)
    I = np.zeros(num_classes)
    R = np.zeros(num_classes)
    # Seed infections proportional to degree class size
    remaining=I0
    for i,k in enumerate(ks):
        n_k = int(Pk[k]*N)
        inf_k = min(remaining, max(1,int(I0*Pk[k])))
        remaining -= inf_k
        S[i] = (n_k - inf_k)/n_k if n_k>0 else 1
        I[i] = inf_k/n_k if n_k>0 else 0
    # time history
    times = [0.0]
    S_hist=[S.dot([Pk[k] for k in ks])]
    E_hist=[E.dot([Pk[k] for k in ks])]
    I_hist=[I.dot([Pk[k] for k in ks])]
    R_hist=[R.dot([Pk[k] for k in ks])]
    k_arr = np.array(ks)
    P_arr = np.array([Pk[k] for k in ks])
    k_mean = (k_arr*P_arr).sum()
    steps = int(T/dt)
    for step in range(steps):
        Theta = (k_arr*I*P_arr).sum()/k_mean
        dS = -beta * k_arr * S * Theta
        dE = beta * k_arr * S * Theta - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        S += dS*dt
        E += dE*dt
        I += dI*dt
        R += dR*dt
        times.append(dt*(step+1))
        S_hist.append((S*P_arr).sum())
        E_hist.append((E*P_arr).sum())
        I_hist.append((I*P_arr).sum())
        R_hist.append((R*P_arr).sum())
    return np.array(times), np.array(S_hist)*N, np.array(E_hist)*N, np.array(I_hist)*N, np.array(R_hist)*N

t_er_d,S_er_d,E_er_d,I_er_d,R_er_d = integrate_dbmf(Pk_er,beta_er)
t_ba_d,S_ba_d,E_ba_d,I_ba_d,R_ba_d = integrate_dbmf(Pk_ba,beta_ba)

import pandas as pd

er_d_df = pd.DataFrame({'time':t_er_d,'S':S_er_d,'E':E_er_d,'I':I_er_d,'R':R_er_d})
ba_d_df = pd.DataFrame({'time':t_ba_d,'S':S_ba_d,'E':E_ba_d,'I':I_ba_d,'R':R_ba_d})

er_d_df.to_csv(os.path.join(output_dir,'results-21.csv'),index=False)
ba_d_df.to_csv(os.path.join(output_dir,'results-22.csv'),index=False)
return_dict={'done':'ok'}
