
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import os, json, math, random, sys

current_dir = os.getcwd()
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

N = 2000
mean_k = 8
p = mean_k/(N-1)
G_er = nx.erdos_renyi_graph(N, p, seed=42)
G_ba = nx.barabasi_albert_graph(N, m=4, seed=42)

# Compute degree moments
deg_er = np.array([d for n,d in G_er.degree()])
deg_ba = np.array([d for n,d in G_ba.degree()])
mean_deg_er = deg_er.mean()
mean_deg2_er = (deg_er**2).mean()
mean_deg_ba = deg_ba.mean()
mean_deg2_ba = (deg_ba**2).mean()

q_er = (mean_deg2_er - mean_deg_er)/mean_deg_er
q_ba = (mean_deg2_ba - mean_deg_ba)/mean_deg_ba

gamma = 1/7  # recovery rate per day
sigma = 1/5  # incubation rate per day
R0_target = 2.5

beta_er = R0_target * gamma / q_er
beta_ba = R0_target * gamma / q_ba

# store degree stats
stats = {
    'mean_deg_er': mean_deg_er,
    'mean_deg2_er': mean_deg2_er,
    'mean_deg_ba': mean_deg_ba,
    'mean_deg2_ba': mean_deg2_ba,
    'beta_er': beta_er,
    'beta_ba': beta_ba,
    'q_er': q_er,
    'q_ba': q_ba
}

# Save networks
sparse.save_npz(os.path.join(output_dir, 'er_network.npz'), nx.to_scipy_sparse_array(G_er, format='csr'))
sparse.save_npz(os.path.join(output_dir, 'ba_network.npz'), nx.to_scipy_sparse_array(G_ba, format='csr'))

# Simple stochastic simulation function
def run_seir_network(G, beta, sigma, gamma, T=180, dt=0.5, I0=10):
    N = len(G)
    S = np.ones(N, dtype=int)
    E = np.zeros(N, dtype=int)
    I = np.zeros(N, dtype=int)
    R = np.zeros(N, dtype=int)
    initial_infected = np.random.choice(N, I0, replace=False)
    S[initial_infected] = 0
    I[initial_infected] = 1

    time_points = [0]
    S_counts = [S.sum()]
    E_counts = [E.sum()]
    I_counts = [I.sum()]
    R_counts = [R.sum()]

    t=0
    while t < T and I.sum()+E.sum() > 0:
        t += dt
        # Infection step
        p_inf = 1 - np.exp(-beta*dt)
        new_E = []
        for node in np.where(I==1)[0]:
            for nbr in G.neighbors(node):
                if S[nbr]==1 and random.random()<p_inf:
                    new_E.append(nbr)
        for nidx in new_E:
            S[nidx]=0
            E[nidx]=1
        # Progress E->I
        p_ei = 1 - np.exp(-sigma*dt)
        E_to_I = np.where((E==1) & (np.random.random(N)<p_ei))[0]
        E[E_to_I]=0
        I[E_to_I]=1
        # Recovery I->R
        p_ir = 1 - np.exp(-gamma*dt)
        I_to_R = np.where((I==1) & (np.random.random(N)<p_ir))[0]
        I[I_to_R]=0
        R[I_to_R]=1

        time_points.append(t)
        S_counts.append(S.sum())
        E_counts.append(E.sum())
        I_counts.append(I.sum())
        R_counts.append(R.sum())
    return np.array(time_points), np.array(S_counts), np.array(E_counts), np.array(I_counts), np.array(R_counts)

# Run 20 simulations for each network and average
nsim=20
def multi_sim(G, beta):
    times=[]
    S_list=[]
    E_list=[]
    I_list=[]
    R_list=[]
    for _ in range(nsim):
        t,S,E,I,R=run_seir_network(G,beta,sigma,gamma)
        times.append(t)
        S_list.append(S)
        E_list.append(E)
        I_list.append(I)
        R_list.append(R)
    # Align time arrays by interpolation to common vector
    t_common = np.arange(0,180.1,0.5)
    S_mat = np.zeros((nsim,len(t_common)))
    E_mat = np.zeros_like(S_mat)
    I_mat = np.zeros_like(S_mat)
    R_mat = np.zeros_like(S_mat)
    for i in range(nsim):
        S_mat[i]=np.interp(t_common,times[i],S_list[i])
        E_mat[i]=np.interp(t_common,times[i],E_list[i])
        I_mat[i]=np.interp(t_common,times[i],I_list[i])
        R_mat[i]=np.interp(t_common,times[i],R_list[i])
    return t_common, S_mat.mean(0), E_mat.mean(0), I_mat.mean(0), R_mat.mean(0)

t_er,S_er,E_er,I_er,R_er = multi_sim(G_er,beta_er)
t_ba,S_ba,E_ba,I_ba,R_ba = multi_sim(G_ba,beta_ba)

import pandas as pd

er_df = pd.DataFrame({'time':t_er,'S':S_er,'E':E_er,'I':I_er,'R':R_er})
ba_df = pd.DataFrame({'time':t_ba,'S':S_ba,'E':E_ba,'I':I_ba,'R':R_ba})

er_df.to_csv(os.path.join(output_dir,'results-11.csv'),index=False)
ba_df.to_csv(os.path.join(output_dir,'results-12.csv'),index=False)

# Return stats
return_dict = {'degree_stats':stats}
