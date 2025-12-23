
# Now use the correct node_labels array. Re-run the full simulation for both scenarios.
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import os

def simulate_threshold_cascade(adj_csr, node_labels, group_seed, threshold_K=2, n_runs=100, frac_systemic=0.5, random_seed=None):
    n = adj_csr.shape[0]
    core_idx = np.where(node_labels == 0)[0]
    peri_idx = np.where(node_labels == 1)[0]
    systemic_bool = []
    frac_failed = []
    tsteps = []
    rng = np.random.default_rng(seed=random_seed)
    for run in range(n_runs):
        if group_seed == 'core':
            seed = rng.choice(core_idx)
        else:
            seed = rng.choice(peri_idx)
        state = np.zeros(n, dtype=int)  # healthy=0, failed=1
        state[seed] = 1
        nstep = 0
        while True:
            previously_failed = state == 1
            failed_neighbors = adj_csr @ previously_failed
            eligible = (state == 0) & (failed_neighbors >= threshold_K)
            if not np.any(eligible):
                break
            state[eligible] = 1
            nstep += 1
        frac_f = state.sum() / n
        frac_failed.append(frac_f)
        is_cascade = int(frac_f > frac_systemic)
        systemic_bool.append(is_cascade)
        tsteps.append(nstep)
    results = pd.DataFrame({
        'run': np.arange(1, n_runs + 1),
        'frac_failed': frac_failed,
        'systemic_failure': systemic_bool,
        'tsteps': tsteps
    })
    return results

# Load network matrix
adj_path = os.path.join(os.getcwd(), 'output', 'coreperiphery-sbm-network.npz')
adj_csr = sparse.load_npz(adj_path)
# Insert correct node_labels (from previous successful parsing)
group_path = os.path.join(os.getcwd(), 'output', 'nodegroups-coreperiphery.txt')
def parse_labels(nodegroup_path):
    labels = []
    with open(nodegroup_path, 'r') as fp:
        for line in fp:
            idx, grp = line.strip().split(',')
            if grp.lower().startswith('core'):
                labels.append(0)
            else:
                labels.append(1)
    return np.array(labels)
node_labels = parse_labels(group_path)

results0 = simulate_threshold_cascade(adj_csr, node_labels, group_seed='core', threshold_K=2, n_runs=100, frac_systemic=0.5, random_seed=42)
results0.to_csv(os.path.join(os.getcwd(), 'output', 'results-10.csv'), index=False)

results1 = simulate_threshold_cascade(adj_csr, node_labels, group_seed='periphery', threshold_K=2, n_runs=100, frac_systemic=0.5, random_seed=43)
results1.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11,5), sharey=True)
ax[0].hist(results0['frac_failed'], bins=20, color='tab:blue', alpha=0.8)
ax[0].set_title('Cascade Extent: Core Seed (100 runs)')
ax[0].set_xlabel('Fraction Failed')
ax[0].set_ylabel('Run Count')
ax[0].axvline(0.5, ls='--', color='red', label='Systemic (>50%)')
ax[1].hist(results1['frac_failed'], bins=20, color='tab:green', alpha=0.8)
ax[1].set_title('Cascade Extent: Periphery Seed (100 runs)')
ax[1].set_xlabel('Fraction Failed')
ax[1].axvline(0.5, ls='--', color='red')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-10.png'))
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-11.png'))
plt.close()

# Save summary stats to one CSV
pd.DataFrame({'scenario':['core_seed','periphery_seed'],
              'cascade_probability':[results0['systemic_failure'].mean(),results1['systemic_failure'].mean()],
              'mean_fraction_failed':[results0['frac_failed'].mean(),results1['frac_failed'].mean()],
              'std_fraction_failed':[results0['frac_failed'].std(),results1['frac_failed'].std()]
             }).to_csv(os.path.join(os.getcwd(),'output','results-summary.csv'),index=False)
