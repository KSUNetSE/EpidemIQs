
# Reasoning for Step 3 (ADT Temporal, Model #3):
# FastGEMF does not support evolving edges, so implement custom stochastic SIR simulation:
# - At each day, activate each node with prob alpha=0.1, they select m=2 partners uniformly
# - For each S–I contact, infection with prob per-edge: beta=1.5 per edge per day (rate, so prob = 1-exp(-beta*dt) with dt=1)
# - Recovery with gamma=0.2 per day (prob per day = 1 - exp(-gamma*dt), dt=1)
# - 5 randomly chosen initial I, rest S, N=1000. 100 realizations, each until I=0 or Tmax=50 days
# - Aggregate time series (mean, 90% CI for S, I, R)
import os
import numpy as np
import pandas as pd

N = 1000
alpha = 0.1
m = 2
beta = 1.5
gamma = 0.2
Tmax = 50
runs = 100
np.random.seed()  # allow true stochasticity

# Probabilities per day (from rates, with dt=1)
p_infect = 1 - np.exp(-beta)
p_recover = 1 - np.exp(-gamma)

history_S = []
history_I = []
history_R = []
maxlen = 0

for run in range(runs):
    # Initial states: 0=S, 1=I, 2=R
    states = np.zeros(N, dtype=int)
    infected = np.random.choice(N, 5, replace=False)
    states[infected] = 1
    S = [np.sum(states==0)]
    I = [np.sum(states==1)]
    R = [np.sum(states==2)]
    t = 0
    while (I[-1] > 0) and (t < Tmax):
        t += 1
        # Step 1: Generate today's contacts (activity-driven)
        active = np.where(np.random.rand(N) < alpha)[0]
        partners = [[] for _ in range(N)]
        for node in active:
            candidates = np.setdiff1d(np.arange(N), [node])
            new_partners = np.random.choice(candidates, m, replace=False)
            for partner in new_partners:
                partners[node].append(partner)
                partners[partner].append(node)  # undirected
        # Step 2: Process new infections
        newI = []
        for node in range(N):
            if states[node] == 0:  # susceptible
                contagious = [p for p in partners[node] if states[p] == 1]
                if len(contagious) > 0:
                    infected_flag = 1 - (1 - p_infect)**len(contagious)
                    if np.random.rand() < infected_flag:
                        newI.append(node)
        # Step 3: Process recoveries
        newR = [i for i in range(N) if (states[i]==1) and (np.random.rand()<p_recover)]
        # Step 4: Update states
        for ni in newI:
            states[ni] = 1
        for nr in newR:
            states[nr] = 2
        S.append(np.sum(states==0))
        I.append(np.sum(states==1))
        R.append(np.sum(states==2))
    # Pad to common time
    if len(S) > maxlen:
        maxlen = len(S)
    history_S.append(S)
    history_I.append(I)
    history_R.append(R)

# Pad all histories
for arr in [history_S, history_I, history_R]:
    for row in arr:
        while len(row) < maxlen:
            row.append(row[-1])

# Convert to arrays
Smat = np.array(history_S)
Imat = np.array(history_I)
Rmat = np.array(history_R)
time = np.arange(maxlen)

# Mean and 90% confidence interval
mean_S = Smat.mean(axis=0)
mean_I = Imat.mean(axis=0)
mean_R = Rmat.mean(axis=0)
ci_lo_S = np.percentile(Smat, 5, axis=0)
ci_hi_S = np.percentile(Smat, 95, axis=0)
ci_lo_I = np.percentile(Imat, 5, axis=0)
ci_hi_I = np.percentile(Imat, 95, axis=0)
ci_lo_R = np.percentile(Rmat, 5, axis=0)
ci_hi_R = np.percentile(Rmat, 95, axis=0)

out = pd.DataFrame({
    'time': time,
    'S': mean_S,
    'I': mean_I,
    'R': mean_R,
    'S_90ci_lower': ci_lo_S, 'S_90ci_upper': ci_hi_S,
    'I_90ci_lower': ci_lo_I, 'I_90ci_upper': ci_hi_I,
    'R_90ci_lower': ci_lo_R, 'R_90ci_upper': ci_hi_R,
})
out.to_csv(os.path.join(os.getcwd(), 'output', 'results-13.csv'), index=False)

# Plot and save
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(time, mean_S, label='Susceptible', color='tab:blue')
plt.fill_between(time, ci_lo_S, ci_hi_S, alpha=0.2, color='tab:blue')
plt.plot(time, mean_I, label='Infectious', color='tab:orange')
plt.fill_between(time, ci_lo_I, ci_hi_I, alpha=0.2, color='tab:orange')
plt.plot(time, mean_R, label='Recovered', color='tab:green')
plt.fill_between(time, ci_lo_R, ci_hi_R, alpha=0.2, color='tab:green')
plt.xlabel('Time (days)')
plt.ylabel('Individuals')
plt.title('SIR on Activity-Driven Temporal Network (N=1000)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-13.png'))
plt.close()
