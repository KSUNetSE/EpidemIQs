
# Observation: AttributeError indicates likely issue with the column names in edge_table: either not 't', 'src', 'tgt' or unexpected header style.
# Step 1: Quickly inspect columns of edge CSV for debugging, print columns so we know the correct usage for main code.
import pandas as pd
edge_table_path = '/Users/hosseinsamaei/phd/epidemiqs/output/temporal-edgetable.csv'
temp_df = pd.read_csv(edge_table_path, nrows=5)
edge_cols = temp_df.columns.tolist()
edge_head = temp_df.head().to_dict()

# Fix in simulation: column is 'time', not 't'. Update accordingly.
# Re-run main temporal network SIR simulation.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

N = 1000
T = 100
per_contact_infect_prob = 0.777
recovery_prob = 0.2
nsim = 100

edge_table_path = '/Users/hosseinsamaei/phd/epidemiqs/output/temporal-edgetable.csv'
edges_by_timestep = defaultdict(list)

temp_df = pd.read_csv(edge_table_path)
for row in temp_df.itertuples():
    t = int(row.time)
    src = int(row.src)
    tgt = int(row.tgt)
    edges_by_timestep[t].append((src, tgt))
del temp_df

results = []
final_sizes = []
peak_prevalences = []
times_to_peak = []
for sim in range(nsim):
    states = np.zeros(N, dtype=np.int8)
    seed = np.random.choice(N)
    states[seed] = 1
    S_count = [N-1]
    I_count = [1]
    R_count = [0]
    for t in range(T):
        new_states = states.copy()
        infectious = np.where(states == 1)[0]
        recover_idx = infectious[np.random.rand(len(infectious)) < recovery_prob]
        new_states[recover_idx] = 2
        contacts = edges_by_timestep.get(t, [])
        for src, tgt in contacts:
            if (states[src] == 1 and states[tgt] == 0):
                if np.random.rand() < per_contact_infect_prob:
                    new_states[tgt] = 1
            elif (states[tgt] == 1 and states[src] == 0):
                if np.random.rand() < per_contact_infect_prob:
                    new_states[src] = 1
        states = new_states
        S_count.append(np.sum(states == 0))
        I_count.append(np.sum(states == 1))
        R_count.append(np.sum(states == 2))
    arr = np.vstack([S_count, I_count, R_count])
    results.append(arr)
    final_sizes.append(R_count[-1]/N)
    peak_prevalences.append(np.max(I_count)/N)
    times_to_peak.append(np.argmax(I_count))
results = np.stack(results, axis=0)

mean_traj = results.mean(axis=0)
lower_traj = np.percentile(results, 5, axis=0)
upper_traj = np.percentile(results, 95, axis=0)
timesteps = np.arange(T+1)

summary = {
    'final_size_mean': np.mean(final_sizes),
    'final_size_5p': np.percentile(final_sizes, 5),
    'final_size_95p': np.percentile(final_sizes, 95),
    'peak_prev_mean': np.mean(peak_prevalences),
    'peak_prev_5p': np.percentile(peak_prevalences, 5),
    'peak_prev_95p': np.percentile(peak_prevalences, 95),
    't_peak_mean': np.mean(times_to_peak),
    't_peak_5p': np.percentile(times_to_peak, 5),
    't_peak_95p': np.percentile(times_to_peak, 95),
}

curdir = os.getcwd()
output_dir = os.path.join(curdir, 'output')
os.makedirs(output_dir, exist_ok=True)
out_df = pd.DataFrame({'time': timesteps,
                       'S_mean': mean_traj[0],
                       'I_mean': mean_traj[1],
                       'R_mean': mean_traj[2],
                       'S_5p': lower_traj[0], 'S_95p': upper_traj[0],
                       'I_5p': lower_traj[1], 'I_95p': upper_traj[1],
                       'R_5p': lower_traj[2], 'R_95p': upper_traj[2]})
for k,v in summary.items():
    out_df[k] = [v] + [None]*T
out_csv_path = os.path.join(curdir, 'output', 'results-11.csv')
out_df.to_csv(out_csv_path, index=False)

plt.figure(figsize=(8,5))
plt.fill_between(timesteps, lower_traj[1], upper_traj[1], color='orangered', alpha=0.15, label='I: 90% CI')
plt.plot(timesteps, mean_traj[0], label='Susceptible (mean)', color='blue', lw=2)
plt.plot(timesteps, mean_traj[1], label='Infected (mean)', color='red', lw=2)
plt.plot(timesteps, mean_traj[2], label='Recovered (mean)', color='green', lw=2)
plt.xlabel('Time step')
plt.ylabel('Population count')
plt.title('SIR on Activity-driven Temporal Network (mean, 90% CI, n=100)')
plt.legend()
out_png_path = os.path.join(curdir, 'output', 'results-11.png')
plt.tight_layout()
plt.savefig(out_png_path)
plt.close()

out_csv_path = out_csv_path
out_png_path = out_png_path
summary = summary
shape_results_array = results.shape
sim_ensemble_final_sizes = final_sizes[:5]
