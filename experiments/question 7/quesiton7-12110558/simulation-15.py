
# --- CHAIN OF THOUGHT Step 4/4 ---
# Now that we have generated a proper network/group file for kc=1.0, rerun the simulation for kc=1.0 for both core/periphery seeds (i=1,2, j=4) and add results.
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

def load_network_and_groups(net_path, group_path):
    A = sp.load_npz(net_path)
    groups = pd.read_csv(group_path)['group'].values
    if len(groups) != A.shape[0]:
        raise ValueError("Mismatch between group labels and adjacency matrix size")
    return A, groups

def threshold_cascade(A, theta, seed_idx):
    N = A.shape[0]
    states = np.zeros(N, dtype=int)
    states[seed_idx] = 1
    changed = True
    while changed:
        changed = False
        to_fail = []
        sus_idx = np.where(states == 0)[0]
        if len(sus_idx) == 0: break
        neighF = A[sus_idx, :].dot(states)
        for idx, nF in zip(sus_idx, neighF):
            if nF >= theta:
                to_fail.append(idx)
        if to_fail:
            states[to_fail] = 1
            changed = True
    return states.sum()

def run_threshold_scenario(A, groups, theta, seed_group, n_reps, frac_global=0.5, random_seed=444):
    np.random.seed(random_seed)
    group_indices = np.where(groups == seed_group)[0]
    results = []
    for rep in range(n_reps):
        seed_idx = np.random.choice(group_indices)
        total_failed = threshold_cascade(A, theta, seed_idx)
        is_global = (total_failed > frac_global * len(groups))
        results.append({
            'seed_idx': seed_idx,
            'final_failed': total_failed,
            'is_global_cascade': int(is_global)
        })
    return pd.DataFrame(results)

kc = 1.0
net_path = os.path.join(os.getcwd(), 'output', 'core-periphery-network-kc1.00.npz')
group_path = os.path.join(os.getcwd(), 'output', 'core-periphery-groups-kc1.00.csv')
A, groups = load_network_and_groups(net_path, group_path)
results_dict = {}
for i, seed_group in enumerate(['core', 'periphery'], start=1):
    df = run_threshold_scenario(A, groups, theta=2, seed_group=seed_group, n_reps=20, random_seed=999+i)
    out_path = os.path.join(os.getcwd(), 'output', f'results-{i}4.csv')
    df.to_csv(out_path, index=False)
    results_dict[out_path] = f"kc=1.00, seed in {seed_group}: per-run final failed, global cascade indicator"
# Also store summary for plotting reference
cascade_probs = [df['is_global_cascade'].mean() for seed_group in ['core', 'periphery']]
_return = {'results_dict': results_dict, 'kc': kc, 'cascade_probs': cascade_probs}
