
# Chain-of-Thought:
# Step 1: Set up simulation parameters (network/ensemble size, thresholds, p_cc values).
# Step 2: Implement a function to generate a core-periphery network (N_core=20, N_periphery=80) given p_cc, p_cp, p_pp.
# Step 3: Implement the synchronous threshold update rule (for state N to F, threshold=2), absorbing dynamics.
# Step 4: Implement repeat simulation logic for two initial conditions -- scenario A (randomly fail 2 core nodes), scenario B (randomly fail 2 periphery nodes).
# Step 5: For each network realization/ensemble, store cascade size, rounds, whether systemic (>20) failures observed.
# Step 6: Aggregate data, save as CSV and PNG plots (fraction systemic per p_cc, etc.).
#
# This code runs for a single p_cc (index i=1) for both scenarios (core/periphery seeding, j=1/j=2), saving results under prescribed filenames.

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- PARAMETERS --- #
N = 100
N_core = 20
N_periphery = 80
threshold = 2
p_cc = 0.5  # Example value, index i=1
p_cp = 0.2
p_pp = 0.02
ensemble_size = 500  # Number of repeats
systemic_thresh = 20  # >20 nodes failed => systemic event
np.random.seed(123)

# Helper: Generate core-periphery network
def generate_core_periphery(N, N_core, N_periphery, p_cc, p_cp, p_pp, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    A = np.zeros((N, N), dtype=int)
    # Core-core
    for i in range(N_core):
        for j in range(i+1, N_core):
            if rng.rand() < p_cc:
                A[i, j] = A[j, i] = 1
    # Core-periphery
    for i in range(N_core):
        for j in range(N_core, N):
            if rng.rand() < p_cp:
                A[i, j] = A[j, i] = 1
    # Periphery-periphery
    for i in range(N_core, N):
        for j in range(i+1, N):
            if rng.rand() < p_pp:
                A[i, j] = A[j, i] = 1
    return sp.csr_matrix(A)

# Helper: Threshold cascade simulation (synchronous)
def simulate_threshold_cascade(adj, initial_state, threshold):
    N = adj.shape[0]
    states = initial_state.copy()
    rounds = 0
    states_hist = [states.copy()]
    while True:
        to_fail = []
        for node in range(N):
            if states[node] == 0:  # N (not failed)
                n_failed = adj[node].dot(states)
                if n_failed >= threshold:
                    to_fail.append(node)
        if not to_fail:
            break
        states[to_fail] = 1  # F (failed)
        states_hist.append(states.copy())
        rounds += 1
    return np.array(states_hist), rounds

# Simulation wrappers for scenario A (core seeding) and B (periphery seeding)
def run_ensemble(adj, threshold, scenario, ensemble_size, systemic_thresh=20, save_path=None):
    results = []
    N = adj.shape[0]
    if scenario == 'core':
        pool = list(range(N_core))
    elif scenario == 'periphery':
        pool = list(range(N_core, N))
    else:
        raise ValueError('Unknown scenario')
    for run in range(ensemble_size):
        states0 = np.zeros(N, dtype=int)
        seed_nodes = np.random.choice(pool, 2, replace=False)
        states0[seed_nodes] = 1  # Set failed
        states_hist, rounds = simulate_threshold_cascade(adj, states0, threshold)
        n_failed = states_hist[-1].sum()
        systemic = n_failed > systemic_thresh
        results.append({'run': run + 1, 'n_final_failed': int(n_failed), 'systemic': systemic, 'rounds': rounds, 'seed_nodes': seed_nodes.tolist()})
    df = pd.DataFrame(results)
    if save_path:
        df.to_csv(save_path, index=False)
    return df

# --- MAIN --- #
# 1. Generate network instance for p_cc = 0.5 (i=1)
adj_csr = generate_core_periphery(N, N_core, N_periphery, p_cc, p_cp, p_pp)

# 2A. Scenario A: fail 2 random core nodes
save_path_A = os.path.join(os.getcwd(), 'output', 'results-11.csv')
df_A = run_ensemble(adj_csr, threshold, 'core', ensemble_size, systemic_thresh, save_path=save_path_A)

# Plot for A
plt.figure(figsize=(6, 4))
n_failed_A = df_A['n_final_failed']
plt.hist(n_failed_A, bins=np.arange(0, N+2, 2), alpha=0.7, color='C0', label='Core shock')
plt.axvline(systemic_thresh, linestyle='--', color='k', label='systemic threshold')
plt.xlabel('Cascade size (final # failed)')
plt.ylabel('Frequency')
plt.title('Cascade size distribution (Core shock, p_cc = 0.5)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-11.png'))
plt.close()

# 2B. Scenario B: fail 2 periphery nodes
save_path_B = os.path.join(os.getcwd(), 'output', 'results-12.csv')
df_B = run_ensemble(adj_csr, threshold, 'periphery', ensemble_size, systemic_thresh, save_path=save_path_B)

# Plot for B
plt.figure(figsize=(6, 4))
n_failed_B = df_B['n_final_failed']
plt.hist(n_failed_B, bins=np.arange(0, N+2, 2), alpha=0.7, color='C1', label='Periphery shock')
plt.axvline(systemic_thresh, linestyle='--', color='k', label='systemic threshold')
plt.xlabel('Cascade size (final # failed)')
plt.ylabel('Frequency')
plt.title('Cascade size distribution (Periphery shock, p_cc = 0.5)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-12.png'))
plt.close()