
# Instead of assuming column names, map columns by position: first (time), second (u), third (v); rename accordingly for clarity.
# Then repeat network parsing and simulation steps with robust column access.
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

N = 1000
beta = 7.5
gamma = 1.0
nsim = 1000
stop_time = 50
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

table_path = '/Users/hosseinsamaei/phd/epidemiqs/output/temporal-edge-table-n1000-alpha01-m2-T1000.csv'
edges_df = pd.read_csv(table_path)
# Guess column mappings:
if len(edges_df.columns) >= 3:
    time_col, u_col, v_col = edges_df.columns[:3]
else:
    raise ValueError(f"Table has too few columns: {edges_df.columns}")
# Rename for code clarity
edges_df = edges_df.rename(columns={time_col: 't', u_col: 'i', v_col: 'j'})

T = min(stop_time, int(edges_df['t'].max()) + 1)
all_time_edges = [[] for _ in range(T)]
for _, row in edges_df.iterrows():
    t = int(row['t'])
    if t < T:
        all_time_edges[t].append((int(row['i']), int(row['j'])))

def run_temporal_sir(nsim, N, T, beta, gamma, all_time_edges):
    S_traj = np.zeros((nsim, T+1))
    I_traj = np.zeros((nsim, T+1))
    R_traj = np.zeros((nsim, T+1))
    for sim in tqdm(range(nsim)):
        state = np.zeros(N, dtype=np.int8)  # 0: S, 1: I, 2: R
        patient_zero = np.random.choice(N,5)
        state[patient_zero] = 1
        st, it, rt = [N-1], [1], [0]
        for t in range(T):
            edges = all_time_edges[t]
            infected = np.where(state == 1)[0]
            susceptible = np.where(state == 0)[0]
            # --- Transmit infection ---
            newly_infected = set()
            # Only check edges that touch I nodes, for efficiency
            for u, v in edges:
                S_side, I_side = None, None
                if state[u] == 0 and state[v] == 1:
                    S_side, I_side = u, v
                elif state[u] == 1 and state[v] == 0:
                    S_side, I_side = v, u
                if S_side is not None:
                    if np.random.uniform() < 1 - np.exp(-beta):
                        newly_infected.add(S_side)
            # Recovery step
            newly_recovered = []
            for i in infected:
                if np.random.uniform() < 1 - np.exp(-gamma):
                    newly_recovered.append(i)
            # Update states
            for i in newly_infected:
                state[i] = 1
            for i in newly_recovered:
                state[i] = 2
            # Count
            st.append(np.sum(state == 0))
            it.append(np.sum(state == 1))
            rt.append(np.sum(state == 2))
        S_traj[sim, :] = st
        I_traj[sim, :] = it
        R_traj[sim, :] = rt
    return S_traj, I_traj, R_traj

np.random.seed(123)
S_traj, I_traj, R_traj = run_temporal_sir(nsim, N, T, beta, gamma, all_time_edges)
time = np.arange(T+1)

# Compute mean and 90% CI bands
lo, hi = 5, 95
S_mean = np.mean(S_traj, axis=0)
I_mean = np.mean(I_traj, axis=0)
R_mean = np.mean(R_traj, axis=0)
S_lo = np.percentile(S_traj, lo, axis=0)
S_hi = np.percentile(S_traj, hi, axis=0)
I_lo = np.percentile(I_traj, lo, axis=0)
I_hi = np.percentile(I_traj, hi, axis=0)
R_lo = np.percentile(R_traj, lo, axis=0)
R_hi = np.percentile(R_traj, hi, axis=0)

out_df = pd.DataFrame({
    'time': time,
    'S': S_mean,     'S_90ci_lower': S_lo, 'S_90ci_upper': S_hi,
    'I': I_mean,     'I_90ci_lower': I_lo, 'I_90ci_upper': I_hi,
    'R': R_mean,     'R_90ci_lower': R_lo, 'R_90ci_upper': R_hi
})
results_path = os.path.join(output_dir, 'results-12.csv')
out_df.to_csv(results_path, index=False)

# Plot result
def plot_temporal_sir(time, S_mean, S_lo, S_hi, I_mean, I_lo, I_hi, R_mean, R_lo, R_hi, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(time, S_mean, label='S', color='tab:blue')
    plt.fill_between(time, S_lo, S_hi, color='tab:blue', alpha=0.2)
    plt.plot(time, I_mean, label='I', color='tab:orange')
    plt.fill_between(time, I_lo, I_hi, color='tab:orange', alpha=0.2)
    plt.plot(time, R_mean, label='R', color='tab:green')
    plt.fill_between(time, R_lo, R_hi, color='tab:green', alpha=0.2)
    plt.xlabel('Time (steps)')
    plt.ylabel('Number of nodes')
    plt.title('Temporal Activity-Driven SIR (N=1000, alpha=0.1, m=2, beta=7.5, gamma=1.0)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

plot_path = os.path.join(output_dir, 'results-12.png')
plot_temporal_sir(time, S_mean, S_lo, S_hi, I_mean, I_lo, I_hi, R_mean, R_lo, R_hi, plot_path)

caption_csv = 'SIR outbreak trajectories on activity-driven temporal network (N=1000, alpha=0.1, m=2, 1000 runs; mean, 90% CI)'
caption_png = 'Epidemic curves Temporal SIR: mean, 90% CI – temporal activity-driven network, R0=3, beta=7.5, gamma=1.0'
