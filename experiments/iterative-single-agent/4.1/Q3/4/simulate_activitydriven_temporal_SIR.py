
# Chain-of-Thought:
# Next, simulate SIR over the activity-driven temporal network. As FastGEMF expects a static network or CSR matrix, not a temporal edge-list,
# we must implement a temporal simulation (simulate the SIR process stepwise on the evolving network snapshots from temporal_edges).
# We will code a stepwise Gillespie-like SIR simulator for a dynamic network. Output will mirror the static case (CSV and PNG for comparison).

import numpy as np
import pandas as pd
import os
from collections import defaultdict

# Setup
N = 1000
T_total = 150    # total time steps for SIR simulation
alpha = 0.1
m = 2
beta = 3.0      # from earlier (temporal SIR beta)
gamma = 0.2     # recovery rate
np.random.seed(42)

# Initial state (1% infected)
X = np.zeros(N, dtype=int) # 0:S, 1:I, 2:R
infected_init = np.random.choice(N, int(0.01*N), replace=False)
X[infected_init] = 1

# Storage
time_l = []
S_l = []
I_l = []
R_l = []

for t in range(T_total):
    # Generate temporal edges for this time step
    activated = np.where(np.random.rand(N) < alpha)[0]
    edges_t = set()
    for i in activated:
        partners = np.random.choice([x for x in range(N) if x != i], m, replace=False)
        for j in partners:
            edge = tuple(sorted([i, j]))
            edges_t.add(edge)
    # Build adjacency as dict
    adj = defaultdict(list)
    for i, j in edges_t:
        adj[i].append(j)
        adj[j].append(i)
    # Infection process: for each S with I neighbor, infected with prob 1 - exp(-beta * dt)
    new_X = X.copy()
    for i in range(N):
        if X[i] == 0:  # S
            for nbr in adj[i]:
                if X[nbr] == 1:
                    if np.random.rand() < (1 - np.exp(-beta)):
                        new_X[i] = 1
                        break
    # Recovery
    for i in range(N):
        if X[i] == 1:
            if np.random.rand() < (1 - np.exp(-gamma)):
                new_X[i] = 2
    X = new_X
    # Record
    S_l.append(np.sum(X == 0))
    I_l.append(np.sum(X == 1))
    R_l.append(np.sum(X == 2))
    time_l.append(t)

# Save results
res_dict = {'time': time_l, 'S': S_l, 'I': I_l, 'R': R_l}
df = pd.DataFrame(res_dict)
csv_path = os.path.join(os.getcwd(), 'output', 'results-1-2.csv')
df.to_csv(csv_path, index=False)
# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(time_l, S_l, label='Susceptible')
plt.plot(time_l, I_l, label='Infected')
plt.plot(time_l, R_l, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR dynamics on activity-driven temporal network')
plt.legend()
plt.tight_layout()
fig_path = os.path.join(os.getcwd(),'output', 'results-1-2.png')
plt.savefig(fig_path)
csv_path, fig_path
