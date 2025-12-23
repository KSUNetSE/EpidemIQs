
# Fallback: since fastgemf may not accept dynamic_edges, let's simulate temporally varying SIR process manually for comparison.
import numpy as np
import pickle
import random
import os
import pandas as pd

N = 1000
gamma = 0.2
beta = 3.0
T_steps = 80 # time steps

# Load temporal edge snapshots
with open(os.path.join(os.getcwd(), 'output', 'temporal_snapshots.pkl'), 'rb') as f:
    edge_snapshots = pickle.load(f)

# Initial state: 1% infected (10 nodes), rest susceptible, 0=S, 1=I, 2=R
state = np.zeros(N, dtype=int)
I_nodes = random.sample(range(N), 10)
state[I_nodes] = 1

time = [0]
S_list = [np.sum(state == 0)]
I_list = [np.sum(state == 1)]
R_list = [np.sum(state == 2)]

# At each time step, update states using current snapshot
for t in range(1, T_steps + 1):
    this_edges = edge_snapshots[t % len(edge_snapshots)]  # re-loop snapshot if needed
    # Build adjacency
    adj = [[] for _ in range(N)]
    for u, v in this_edges:
        adj[u].append(v)
        adj[v].append(u)
    new_state = state.copy()
    # Infect S -> I if at least one neighbor is I, by transmission per contact
    for i in range(N):
        if state[i] == 0:  # susceptible
            infected = [j for j in adj[i] if state[j] == 1]
            # probability not to get infected = (1-beta)^(#infected contacts)
            p_infect = 1 - (1 - beta) ** len(infected)
            if random.random() < p_infect and len(infected) > 0:
                new_state[i] = 1
        elif state[i] == 1:  # infected
            # Recovery
            if random.random() < gamma:
                new_state[i] = 2
    state = new_state
    time.append(t)
    S_list.append(np.sum(state == 0))
    I_list.append(np.sum(state == 1))
    R_list.append(np.sum(state == 2))

# Save CSV as the static case for comparison
sim_results = {'time': time, 'S': S_list, 'I': I_list, 'R': R_list}
df = pd.DataFrame(sim_results)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
