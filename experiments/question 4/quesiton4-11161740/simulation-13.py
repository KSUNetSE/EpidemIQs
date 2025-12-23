
# Scenario 3: High-overlap, param set 3 (i=1, j=3)
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
np.random.seed(44)
N = 1000
beta1 = 0.05482
delta1 = 1.0
beta2 = 0.06122
delta2 = 1.0
Tmax = 500.0
nsim = 150
t_A = os.path.join(os.getcwd(),'output','layerA-highoverlap.npz')
t_B = os.path.join(os.getcwd(),'output','layerB-highoverlap.npz')
A = sp.load_npz(t_A)
B = sp.load_npz(t_B)
def run_one_realization():
    states = np.zeros(N, dtype=int)
    idx_all = np.random.permutation(N)
    nI1 = int(N*0.05)
    nI2 = int(N*0.05)
    states[idx_all[:nI1]] = 1
    states[idx_all[nI1:nI1+nI2]] = 2
    t_record = [0.0]
    S_record = [np.sum(states == 0)]
    I1_record = [np.sum(states == 1)]
    I2_record = [np.sum(states == 2)]
    t = 0.0
    while t < Tmax:
        infected_A = (states == 1).astype(float)
        infected_B = (states == 2).astype(float)
        nI1_neighbors = A @ infected_A
        nI2_neighbors = B @ infected_B
        susceptible = (states == 0)
        rate_inf1 = beta1 * nI1_neighbors * susceptible
        rate_inf2 = beta2 * nI2_neighbors * susceptible
        rate_rec1 = delta1 * (states == 1)
        rate_rec2 = delta2 * (states == 2)
        rates = np.concatenate([rate_inf1, rate_inf2, rate_rec1, rate_rec2])
        total_rate = np.sum(rates)
        if total_rate == 0:
            t = Tmax
            t_record.append(t)
            S_record.append(np.sum(states == 0))
            I1_record.append(np.sum(states == 1))
            I2_record.append(np.sum(states == 2))
            break
        dt = np.random.exponential(1/total_rate)
        t += dt
        event_idx = np.random.choice(4*N, p=rates/total_rate)
        if event_idx < N:
            i = event_idx
            if states[i] == 0:
                states[i] = 1
        elif event_idx < 2*N:
            i = event_idx - N
            if states[i] == 0:
                states[i] = 2
        elif event_idx < 3*N:
            i = event_idx - 2*N
            if states[i] == 1:
                states[i] = 0
        else:
            i = event_idx - 3*N
            if states[i] == 2:
                states[i] = 0
        if len(t_record) == 1 or t > t_record[-1] + 0.5:
            t_record.append(t)
            S_record.append(np.sum(states == 0))
            I1_record.append(np.sum(states == 1))
            I2_record.append(np.sum(states == 2))
    time_grid = np.arange(0, Tmax+0.5, 0.5)
    S_interp = np.interp(time_grid, t_record, S_record)
    I1_interp = np.interp(time_grid, t_record, I1_record)
    I2_interp = np.interp(time_grid, t_record, I2_record)
    return time_grid, S_interp, I1_interp, I2_interp
all_S = []
all_I1 = []
all_I2 = []
for rep in range(nsim):
    _, S_c, I1_c, I2_c = run_one_realization()
    all_S.append(S_c)
    all_I1.append(I1_c)
    all_I2.append(I2_c)
all_S = np.array(all_S)
all_I1 = np.array(all_I1)
all_I2 = np.array(all_I2)
time_grid = np.arange(0, Tmax+0.5, 0.5)
S_mean = np.mean(all_S, axis=0)/N
I1_mean = np.mean(all_I1, axis=0)/N
I2_mean = np.mean(all_I2, axis=0)/N
S_low = np.percentile(all_S/N, 5, axis=0)
S_up = np.percentile(all_S/N, 95, axis=0)
I1_low = np.percentile(all_I1/N, 5, axis=0)
I1_up = np.percentile(all_I1/N, 95, axis=0)
I2_low = np.percentile(all_I2/N, 5, axis=0)
I2_up = np.percentile(all_I2/N, 95, axis=0)
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
output_csv = os.path.join(os.getcwd(),'output','results-13.csv')
result_df.to_csv(output_csv, index=False)
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
output_png = os.path.join(os.getcwd(),'output','results-13.png')
plt.savefig(output_png)
plt.close()