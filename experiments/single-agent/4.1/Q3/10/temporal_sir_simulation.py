
# Since FastGEMF seems to expect a static or multilayer (but not rapidly changing temporal layer) structure, and succeeds only for single csr_matrix static network,
# For the temporal simulation, summarize the experiment by hand-coding the SIR evolution at each time step based on the time-varying edge list
# Let's implement a simple event-driven SIR simulation for the temporal network-edge_lists, consistent with Gillespie logic
import numpy as np
import pandas as pd
import os
import random
edge_lists_per_t = np.load(os.path.join(os.getcwd(), 'output', 'temporal_network_edges.npz'), allow_pickle=True)
T = len(edge_lists_per_t.files)
N = 1000
beta = 0.0018239  # static value; temporal transmission probability will be scaled by dt=1 per contact

# SIR params
infectious_period = 5  # <-> gamma=0.2
nsim = 1

summary_time = []
summary_S = []
summary_I = []
summary_R = []
for simid in range(nsim):
    states = np.zeros(N, dtype=int)  # 0=S, 1=I, 2=R
    infected = np.random.choice(N, 5, replace=False)
    states[infected] = 1
    recov_time = np.full(N, np.inf)  # time each node recovers

    time_series = []
    for t in range(T):
        time_series.append((t, (states==0).sum(), (states==1).sum(), (states==2).sum()))
        edges = edge_lists_per_t[f'arr_{t}']
        for (i,j) in edges:
            for u,v in [(i,j),(j,i)]:  # undirected; try both ways
                if states[u]==1 and states[v]==0:
                    # try infection
                    if random.random() < beta:  # beta is per susceptible-infected contact per step
                        states[v]=1
                        recov_time[v] = t + infectious_period
        # recover anyone whose time is up
        for node in range(N):
            if states[node]==1 and t >= recov_time[node]:
                states[node]=2
    # Record last
    time_series.append((T, (states==0).sum(), (states==1).sum(), (states==2).sum()))
    df = pd.DataFrame(time_series, columns=['time','S','I','R'])
    df.to_csv(os.path.join(os.getcwd(), 'output', f'results-12.csv'), index=False)
'temporal_sir_done'