
import os, numpy as np, random, pandas as pd, scipy.sparse as sparse, networkx as nx, math, matplotlib.pyplot as plt
from collections import defaultdict

# Parameters
N = 10000
initial_infected = 100  # 1%
S=0; I=1; R=2

gamma = 0.2  # recovery rate per day (continuous). convert to discrete probability approximate: p_rec = 1-exp(-gamma)
p_rec = 1 - math.exp(-gamma)

# Activity-driven params
xmin = 1e-3
alpha = 2.5  # power-law exponent
# sample activity using inverse transform
rand = np.random.random(N)
activity = xmin * (1 - rand) ** (-1/(alpha-1))
activity[activity>1]=1  # cap at 1
m = 3  # number of links per activation
# compute mean and mean square
mean_a = activity.mean()
mean_a2 = (activity**2).mean()
Delta = mean_a + math.sqrt(mean_a2)

# compute beta_temporal s.t R0=3
R0_target = 3
beta_temp = R0_target * gamma / (m * Delta)

# discrete-time infection probability per contact approximate p=1-exp(-beta) with dt=1
p_inf_temporal = 1 - math.exp(-beta_temp)
# but if beta_temp small, approximate as beta_temp.

print('Temporal beta', beta_temp, 'p_inf', p_inf_temporal, 'mean_a', mean_a, 'mean_a2', mean_a2)

# Simulation parameters
T_max = 200  # days
simulations = 10

time_series_S = np.zeros(T_max+1)
time_series_I = np.zeros(T_max+1)
time_series_R = np.zeros(T_max+1)

for sim in range(simulations):
    states = np.zeros(N, dtype=int)
    infected_indices = np.random.choice(N, initial_infected, replace=False)
    states[infected_indices] = I
    time_series_S[0] += (N-initial_infected)
    time_series_I[0] += initial_infected
    time_series_R[0] += 0
    for t in range(1, T_max+1):
        # generate contacts for this step
        contacts = []
        # nodes that activate
        active = np.where(np.random.random(N) < activity)[0]
        for u in active:
            targets = np.random.choice(N, m, replace=False)
            for v in targets:
                contacts.append((u, v))
        # infection step
        new_infected = []
        for u,v in contacts:
            if states[u]==I and states[v]==S and np.random.random() < p_inf_temporal:
                new_infected.append(v)
            elif states[v]==I and states[u]==S and np.random.random() < p_inf_temporal:
                new_infected.append(u)
        states[new_infected] = I
        # recovery
        recovering = np.where((states==I) & (np.random.random(N) < p_rec))[0]
        states[recovering] = R
        # record
        time_series_S[t] += np.sum(states==S)
        time_series_I[t] += np.sum(states==I)
        time_series_R[t] += np.sum(states==R)
        # early stop if no infected
        if np.sum(states==I)==0:
            # fill remaining with same counts
            for tt in range(t+1, T_max+1):
                time_series_S[tt] += np.sum(states==S)
                time_series_I[tt] += 0
                time_series_R[tt] += np.sum(states==R)
            break

# average over simulations
time_series_S /= simulations
time_series_I /= simulations
time_series_R /= simulations

data = pd.DataFrame({'time':np.arange(T_max+1), 'S':time_series_S, 'I':time_series_I, 'R':time_series_R})

output_dir = os.path.join(os.getcwd(),'output')
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir,'results-12.csv')
data.to_csv(csv_path,index=False)

# plot
plt.figure()
plt.plot(data['time'], data['S'], label='S')
plt.plot(data['time'], data['I'], label='I')
plt.plot(data['time'], data['R'], label='R')
plt.xlabel('Time (days)')
plt.ylabel('Individuals')
plt.legend()
plt.title('Activity-driven Temporal Network Epidemic (average over 10 runs)')
plot_path = os.path.join(output_dir,'results-12.png')
plt.savefig(plot_path)
plt.close()

csv_path, plot_path