
# Step 2: Activity-driven temporal SIR simulation (Scenario 2, i=1, j=2)
# Plan: Load temporal edgelist, run agent-based synchronous SIR simulation. Not suitable for FastGEMF, so we implement custom code. Save CSV/plot as required.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
N = 1000
beta = 0.5
gamma = 1.0
T = 100  # simulation time = given temporal event table
n_sim = 100
seed_value = 42
results_csv = os.path.join(os.getcwd(), 'output/EpidemIQs-4.1/Q6/4', 'new-temporal.csv')
results_png = os.path.join(os.getcwd(), 'output/EpidemIQs-4.1/Q6/4', 'new-temporal.png')
edgelist_path = os.path.join(os.getcwd(), 'output/EpidemIQs-4.1/Q6/4', 'activity-driven-temporal-edgelist.csv')

np.random.seed(seed_value)

# 1. Load temporal edge list (assumed columns: time, node_i, node_j)
edgelist_df = pd.read_csv(edgelist_path)
assert all(c in edgelist_df.columns for c in ['time', 'node_i', 'node_j'])
Tmax = int(edgelist_df['time'].max())
assert Tmax == T - 1 or Tmax == T

# Generate list of edges per timestep
dt_contacts = [[] for _ in range(1000)]
for _, row in edgelist_df.iterrows():
    dt_contacts[int(row['time'])].append((int(row['node_i']), int(row['node_j'])))

# 2. Prepare arrays to accumulate S/I/R counts each time step over realizations
S_all = np.zeros((n_sim, T + 1))
I_all = np.zeros((n_sim, T + 1))
R_all = np.zeros((n_sim, T + 1))
time=np.linspace(0, T, num=1000)
for sim_num in range(n_sim):
    # Seed initial states
    X = np.zeros(N, dtype=np.int8)  # 0=S, 1=I, 2=R
    initial_infected = np.random.choice(N, size=5, replace=False)
    X[initial_infected] = 1
    # Trajectory record
    S_traj = [np.sum(X == 0)]
    I_traj = [np.sum(X == 1)]
    R_traj = [np.sum(X == 2)]
    for idx,t in enumerate(time): #range(T):
        # -- Recovery: for all I, recover with probability p = 1 - exp(-gamma*dt) ~ gamma*dt for small dt=1
        I_idx, = np.where(X == 1)
        recov_prob = 1 - np.exp(-gamma)  # dt=1
        recover = np.random.rand(len(I_idx)) < recov_prob
        X[I_idx[recover]] = 2

        # -- Infection: only possible via current edges
        contacts = dt_contacts[idx]
        for (i, j) in contacts:
            # Only S-I pairs, can infect each other!
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

# Average and save time series
S_mean = S_all.mean(axis=0)
I_mean = I_all.mean(axis=0)
R_mean = R_all.mean(axis=0)
results = pd.DataFrame({'time': np.arange(T + 1), 'S': S_mean, 'I': I_mean, 'R': R_mean})
results.to_csv(results_csv, index=False)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(results['time'], results['S'], label='Susceptible', color='tab:blue')
plt.plot(results['time'], results['I'], label='Infected', color='tab:orange')
plt.plot(results['time'], results['R'], label='Recovered', color='tab:green')
plt.xlabel('Time step')
plt.ylabel('Population')
plt.title('SIR on Activity-Driven Temporal Network (beta=0.5, gamma=1.0, N=1000)')
plt.legend()
plt.tight_layout()
plt.savefig(results_png)
plt.close()

net_nodes = N
unique_contacts = set((min(i, j), max(i, j)) for contacts in dt_contacts for (i, j) in contacts)
net_edges = len(unique_contacts)
(net_nodes, net_edges)
