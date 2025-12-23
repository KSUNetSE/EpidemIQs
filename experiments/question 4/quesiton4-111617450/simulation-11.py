
# Reasoning: The problem requires simulating a mutually exclusive competitive SIS epidemic on a multiplex network using explicit Gillespie-style simulation logic because FastGEMF does not natively handle mutually exclusive compartmental models involving simultaneous, exclusive infection by two agents. The first scenario will be the HIGH-overlap multiplex (dominance structure), for parameter set #1 (i=1, j=1).
# Steps for this simulation:
#  1. Load the adjacency matrices for high-overlap Layer A and Layer B
#  2. Set up the mechanistic discrete-state stochastic process for S-I1-I2 exclusive states.
#  3. Initialize states: 5% I1, 5% I2, 90% S (random, exclusive)
#  4. Simulate many stochastic realizations (nsim=150 recommended per the literature)
#  5. Collect, average, and save prevalence time series and confidence intervals.
#  6. Plot the results.
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import trange
np.random.seed(42)

# PARAMETERS (scenario 1, high-overlap, parameter set 1)
N = 1000
beta1 = 0.04616  # infection rate virus 1 (layer A)
delta1 = 1.0     # recovery rate virus 1
beta2 = 0.04665  # infection rate virus 2 (layer B)
delta2 = 1.0     # recovery rate virus 2
Tmax = 500.0     # max simulation time
nsim = 150       # stochastic realizations

t_A = os.path.join(os.getcwd(),'output','layerA-highoverlap.npz')
t_B = os.path.join(os.getcwd(),'output','layerB-highoverlap.npz')
A = sp.load_npz(t_A)
B = sp.load_npz(t_B)

# Exclusive states: 0 = S, 1 = I1, 2 = I2

def run_one_realization():
    # Initial condition: 5% I1, 5% I2, 90% S, symmetrical, randomly assigned & exclusive
    states = np.zeros(N, dtype=int)
    idx_all = np.random.permutation(N)
    nI1 = int(N*0.05)
    nI2 = int(N*0.05)
    states[idx_all[:nI1]] = 1  # I1
    states[idx_all[nI1:nI1+nI2]] = 2  # I2
    # Schedule arrays
    t_record = [0.0]
    S_record = [np.sum(states == 0)]
    I1_record = [np.sum(states == 1)]
    I2_record = [np.sum(states == 2)]
    t = 0.0
    while t < Tmax:
        # Calculate propensities for each node
        infected_A = (states == 1).astype(float)
        infected_B = (states == 2).astype(float)
        nI1_neighbors = A @ infected_A   # For each node: #I1 neighbors (layer A)
        nI2_neighbors = B @ infected_B   # ... I2 neighbors (layer B)
        # INFECTIONS
        susceptible = (states == 0)
        rate_inf1 = beta1 * nI1_neighbors * susceptible  # Only S can become I1
        rate_inf2 = beta2 * nI2_neighbors * susceptible  # Only S can become I2
        # RECOVERIES
        rate_rec1 = delta1 * (states == 1)  # Only I1 can recover
        rate_rec2 = delta2 * (states == 2)  # Only I2 can recover
        rates = np.concatenate([rate_inf1, rate_inf2, rate_rec1, rate_rec2])
        total_rate = np.sum(rates)
        if total_rate == 0:
            # absorbant state (all S), no further events
            t = Tmax
            t_record.append(t)
            S_record.append(np.sum(states == 0))
            I1_record.append(np.sum(states == 1))
            I2_record.append(np.sum(states == 2))
            break
        # Gillespie step
        dt = np.random.exponential(1/total_rate)
        t += dt
        event_idx = np.random.choice(4*N, p=rates/total_rate)
        # Decode which kind of event
        if event_idx < N:
            # Infection I1 event
            i = event_idx
            if states[i] == 0:
                states[i] = 1
        elif event_idx < 2*N:
            # Infection I2 event
            i = event_idx - N
            if states[i] == 0:
                states[i] = 2
        elif event_idx < 3*N:
            # Recovery I1
            i = event_idx - 2*N
            if states[i] == 1:
                states[i] = 0
        else:
            # Recovery I2
            i = event_idx - 3*N
            if states[i] == 2:
                states[i] = 0
        # Record at regular intervals (to fixed time grid for ensemble average)
        if len(t_record) == 1 or t > t_record[-1] + 0.5:
            t_record.append(t)
            S_record.append(np.sum(states == 0))
            I1_record.append(np.sum(states == 1))
            I2_record.append(np.sum(states == 2))
    # Interpolate counts to fixed time points (dt=0.5)
    time_grid = np.arange(0, Tmax+0.5, 0.5)
    S_interp = np.interp(time_grid, t_record, S_record)
    I1_interp = np.interp(time_grid, t_record, I1_record)
    I2_interp = np.interp(time_grid, t_record, I2_record)
    return time_grid, S_interp, I1_interp, I2_interp

# --- Main: Ensemble Simulation ---
all_S = []
all_I1 = []
all_I2 = []
for rep in trange(nsim, desc='Simulating stochastic realizations'):
    _, S_c, I1_c, I2_c = run_one_realization()
    all_S.append(S_c)
    all_I1.append(I1_c)
    all_I2.append(I2_c)

all_S = np.array(all_S)
all_I1 = np.array(all_I1)
all_I2 = np.array(all_I2)
time_grid = np.arange(0, Tmax+0.5, 0.5)

# Means and 90% confidence band
S_mean = np.mean(all_S, axis=0)/N
I1_mean = np.mean(all_I1, axis=0)/N
I2_mean = np.mean(all_I2, axis=0)/N
S_low = np.percentile(all_S/N, 5, axis=0)
S_up = np.percentile(all_S/N, 95, axis=0)
I1_low = np.percentile(all_I1/N, 5, axis=0)
I1_up = np.percentile(all_I1/N, 95, axis=0)
I2_low = np.percentile(all_I2/N, 5, axis=0)
I2_up = np.percentile(all_I2/N, 95, axis=0)

# Save results to CSV
result_df = pd.DataFrame({
    'time': time_grid,
    'S_mean': S_mean,
    'I1_mean': I1_mean,
    'I2_mean': I2_mean,
    'S_90ci_lower': S_low,
    'S_90ci_upper': S_up,
    'I1_90ci_lower': I1_low,
    'I1_90ci_upper': I1_up,
    'I2_90ci_lower': I2_low,
    'I2_90ci_upper': I2_up
})
output_csv = os.path.join(os.getcwd(),'output','results-11.csv')
result_df.to_csv(output_csv, index=False)

# Plot
plt.figure(figsize=(7,5))
plt.plot(time_grid, S_mean, label='Susceptible', color='tab:blue')
plt.plot(time_grid, I1_mean, label='I1 (Virus 1)', color='tab:orange')
plt.plot(time_grid, I2_mean, label='I2 (Virus 2)', color='tab:green')
plt.fill_between(time_grid, I1_low, I1_up, color='tab:orange', alpha=0.2)
plt.fill_between(time_grid, I2_low, I2_up, color='tab:green', alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Fraction of population')
plt.title('Competitive SIS, High-Overlap, β₁=%.5f, β₂=%.5f' % (beta1, beta2))
plt.legend()
plt.tight_layout()
output_png = os.path.join(os.getcwd(),'output','results-11.png')
plt.savefig(output_png)
plt.close()