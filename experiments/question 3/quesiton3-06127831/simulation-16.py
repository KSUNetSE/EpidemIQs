
# Step 6: Temporal, j=6, beta well above threshold beta=0.7, gamma=1.0 (i=1, j=6)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
N = 1000
beta = 0.7
gamma = 1.0
T = 100
n_sim = 50
seed_value = 123
np.random.seed(seed_value)
results_csv = os.path.join(os.getcwd(), 'output', 'results-16.csv')
results_png = os.path.join(os.getcwd(), 'output', 'results-16.png')
edgelist_path = os.path.join(os.getcwd(), 'output', 'activity-driven-temporal-edgelist.csv')
edgelist_df = pd.read_csv(edgelist_path)
Tmax = int(edgelist_df['time'].max())
assert Tmax == T - 1 or Tmax == T
# Contacts map
dt_contacts = [[] for _ in range(T)]
for _, row in edgelist_df.iterrows():
    dt_contacts[int(row['time'])].append((int(row['node_i']), int(row['node_j'])))
S_all = np.zeros((n_sim, T + 1))
I_all = np.zeros((n_sim, T + 1))
R_all = np.zeros((n_sim, T + 1))
for sim_num in range(n_sim):
    X = np.zeros(N, dtype=np.int8)
    initial_infected = np.random.choice(N, size=5, replace=False)
    X[initial_infected] = 1
    S_traj = [np.sum(X == 0)]
    I_traj = [np.sum(X == 1)]
    R_traj = [np.sum(X == 2)]
    for t in range(T):
        # Recovery
        I_idx, = np.where(X == 1)
        recov_prob = 1 - np.exp(-gamma)
        recover = np.random.rand(len(I_idx)) < recov_prob
        X[I_idx[recover]] = 2
        contacts = dt_contacts[t]
        for (i, j) in contacts:
            for src, tgt in [(i, j), (j, i)]:
                if X[src] == 1 and X[tgt] == 0:
                    inf_prob = 1 - np.exp(-beta)
                    if np.random.rand() < inf_prob:
                        X[tgt] = 1
        S_traj.append(np.sum(X == 0))
        I_traj.append(np.sum(X == 1))
        R_traj.append(np.sum(X == 2))
    S_all[sim_num, :] = S_traj
    I_all[sim_num, :] = I_traj
    R_all[sim_num, :] = R_traj
S_mean = S_all.mean(axis=0)
I_mean = I_all.mean(axis=0)
R_mean = R_all.mean(axis=0)
results = pd.DataFrame({'time': np.arange(T + 1), 'S': S_mean, 'I': I_mean, 'R': R_mean})
results.to_csv(results_csv, index=False)
plt.figure(figsize=(8, 6))
plt.plot(results['time'], results['S'], label='Susceptible', color='tab:blue')
plt.plot(results['time'], results['I'], label='Infected', color='tab:orange')
plt.plot(results['time'], results['R'], label='Recovered', color='tab:green')
plt.xlabel('Time step')
plt.ylabel('Population')
plt.title('SIR on Activity-Driven Temporal Network (beta=0.7)')
plt.legend()
plt.tight_layout()
plt.savefig(results_png)
plt.close()
# Summary info
unique_contacts = set((min(i, j), max(i, j)) for contacts in dt_contacts for (i, j) in contacts)
net_nodes = N
net_edges = len(unique_contacts)
(net_nodes, net_edges)