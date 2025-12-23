
# --- CHAIN OF THOUGHT EXECUTION: Step 1/4 ---
# Step 1: Test/Implement a single deterministic threshold cascade on the supplied network (kc=0.9) with one core-seeded and one periphery-seeded run (10 repetitions for quick check)
# This will validate the core simulation logic and result writing for two scenarios under kc=0.9, before sweeping kc.

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Parameters
net_path = '/Users/hosseinsamaei/phd/epidemiqs/output/core-periphery-network-kc0.90.npz'
group_path = '/Users/hosseinsamaei/phd/epidemiqs/output/core-periphery-groups-kc0.90.csv'
theta = 2
n_reps = 10  # For fast check; increase later as needed
N = 100  # nodes

def load_network_and_groups(net_path, group_path):
    A = sp.load_npz(net_path)
    groups = pd.read_csv(group_path)
    groups = groups['group'].values  # "core" or "periphery"
    if len(groups) != A.shape[0]:
        raise ValueError("Mismatch between group labels and adjacency matrix size")
    return A, groups

# Discrete-time synchronous THRESHOLD CASCADE
# Returns: total_failed (int), time_series (list)
def threshold_cascade(A, theta, seed_idx):
    N = A.shape[0]
    # 0 = S, 1 = F
    states = np.zeros(N, dtype=int)
    states[seed_idx] = 1
    time_series = [states.sum()]
    changed = True
    while changed:
        changed = False
        to_fail = []
        sus_idx = np.where(states == 0)[0]
        if len(sus_idx) == 0:
            break
        neighF = A[sus_idx, :].dot(states)  # for each susceptible: count failed
        for idx, nF in zip(sus_idx, neighF):
            if nF >= theta:
                to_fail.append(idx)
        if to_fail:
            states[to_fail] = 1
            changed = True
        time_series.append(states.sum())
    return states.sum(), time_series

# Simulation wrapper for a single scenario
# seed_group: "core" or "periphery", n_reps: int
# Returns: list of (final_failed_count, is_global_cascade, time_series)
def run_threshold_scenario(A, groups, theta, seed_group, n_reps, frac_global=0.5, random_seed=0):
    np.random.seed(random_seed)
    group_indices = np.where(groups == seed_group)[0]
    results = []
    for rep in range(n_reps):
        seed_idx = np.random.choice(group_indices)
        total_failed, series = threshold_cascade(A, theta, seed_idx)
        is_global = (total_failed > frac_global * len(groups))
        results.append({
            'seed_idx': seed_idx,
            'final_failed': total_failed,
            'is_global_cascade': int(is_global),
            'timesteps': len(series) - 1
        })
    return pd.DataFrame(results)

# Load data
A, groups = load_network_and_groups(net_path, group_path)

results_dict = {}
plot_data = {}

# --- RUN SIMULATION: core-seeded (i=1, j=1), periphery-seeded (i=2, j=1) ---
for i, seed_group in enumerate(['core', 'periphery'], start=1):
    df = run_threshold_scenario(A, groups, theta, seed_group, n_reps, random_seed=888+i)
    out_path = os.path.join(os.getcwd(), 'output', f'results-{i}1.csv')
    df.to_csv(out_path, index=False)
    results_dict[out_path] = f"kc=0.90, seed in {seed_group}: per-run final failed, global cascade indicator"
    # Record for plotting (global cascade freq)
    plot_data[seed_group] = df['is_global_cascade'].mean()

# --- PLOT: Global cascade probability for kc=0.90, both seeding scenarios (core, periphery) ---
import matplotlib.pyplot as plt
plt.figure(figsize=(5,3))
plt.bar(['core', 'periphery'], [plot_data['core'], plot_data['periphery']], color=['red', 'blue'])
plt.ylim(0,1.05)
plt.ylabel("Probability of Global Cascade (>50% failed)")
plt.title("Global Cascade Probability for kc=0.90 (theta=2)")
plot_path = os.path.join(os.getcwd(), 'output', 'results-11.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

# Output
_return = {
    'results_dict': results_dict,
    'plot_path': plot_path
}
