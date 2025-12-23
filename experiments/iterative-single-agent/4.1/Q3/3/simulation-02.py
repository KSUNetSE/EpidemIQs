
# Chain of thought:
# Now we need to simulate the SIR process on the temporal activity-driven network (using the actual temporal edges from activity)
# We'll step through the edge lists, at each step updating states, and tracking counts. We'll use the same beta and gamma, and initial condition as before.
# We'll simulate for 'T' steps, at each step updating according to the temporal edge set for that step.
# We'll store state counts (S, I, R) at each step for plotting and csv.

import numpy as np
import pickle
import os
import pandas as pd

T = 500 # number of steps
n = 1000

gamma = 1/7 # per time step
R0 = 3
k1 = 181.464  # mean deg per aggregate
beta = R0 * gamma / k1  # same calibration (for comparison)

with open(os.path.join(os.getcwd(), 'output', 'temporal_edges.pkl'), 'rb') as f:
    edges_over_time = pickle.load(f)

# Initial condition: 1% infected randomly
num_infected = int(0.01 * n)
init_state = np.zeros(n, dtype=int)
init_state[:num_infected] = 1
np.random.shuffle(init_state)

# State: 0=S, 1=I, 2=R
state_hist = []
cur_state = init_state.copy()

for t, edge_list in enumerate(edges_over_time[:T]):
    # Infection step (S->I) for all edges where i is S, j is I, or vice versa
    new_I = set()
    for i, j in edge_list:
        # Infect i from j
        if cur_state[i] == 0 and cur_state[j] == 1 and np.random.rand() < beta:
            new_I.add(i)
        elif cur_state[j] == 0 and cur_state[i] == 1 and np.random.rand() < beta:
            new_I.add(j)
    # Recovery step (I->R)
    to_R = np.where(cur_state == 1)[0]
    recov = to_R[np.random.rand(len(to_R)) < gamma]
    cur_state[list(recov)] = 2
    # Add new infections
    cur_state[list(new_I)] = 1
    # Store counts
    S = np.sum(cur_state == 0)
    I = np.sum(cur_state == 1)
    R = np.sum(cur_state == 2)
    state_hist.append((S,I,R))

# Save to csv for analysis
state_hist_arr = np.array(state_hist)
df = pd.DataFrame(state_hist_arr, columns=['S','I','R'])
df['time_step'] = np.arange(1, T+1)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-02.csv'), index=False)

return_vars = ['state_hist_arr']
