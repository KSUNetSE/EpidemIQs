
# simulation-01.py: Temporal Activity-Driven SIR, beta=1.55763, gamma=0.1, main case
def run_temporal_activity_driven_sir(
    edge_path, activity_path, N, beta, gamma, m, Tmax, n_sim, output_csv, output_png
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from tqdm import tqdm
    import random

    # 1. Load node activity (N-vector)
    activity = np.load(activity_path)
    assert len(activity) == N, f"Activity vector wrong shape: {activity.shape} vs {N}"
    mean_activity = np.mean(activity)

    # 2. Load temporal contact data: CSV with cols [t, i, j] for each (timestep, node1, node2)
    edge_df = pd.read_csv(edge_path)
    edge_df['t'] = edge_df['t'].astype(int)
    edge_df['i'] = edge_df['i'].astype(int)
    edge_df['j'] = edge_df['j'].astype(int)
    Tmax_actual = edge_df['t'].max() + 1
    assert Tmax_actual >= Tmax, "Not enough timesteps in edge data!"

    # 3. Index edges by time: dict t -> list[(i,j)]  (store as set for fast lookup)
    contacts_by_t = {}
    for t in range(Tmax):
        contacts_by_t[t] = []
    for (t, i, j) in edge_df[['t', 'i', 'j']].itertuples(index=False):
        contacts_by_t[t].append((i, j))

    # 4. Function to run a single stochastic SIR realization over temporal network
    def one_sir_run(seed):
        random.seed(seed)
        np.random.seed(seed)
        # Status: 0=S, 1=I, 2=R
        X = np.zeros(N, dtype=int)
        # Initial: 10 random infected
        infected_init = np.random.choice(N, 10, replace=False)
        X[infected_init] = 1
        S_t, I_t, R_t = [], [], []
        time_to_peak = None
        peak_I = 0
        for t in range(Tmax):
            S_t.append(np.sum(X==0))
            I_t.append(np.sum(X==1))
            R_t.append(np.sum(X==2))
            if I_t[-1] > peak_I:
                peak_I = I_t[-1]
                time_to_peak = t
            # End if no more Infectious
            if I_t[-1]==0:
                break
            this_contacts = contacts_by_t[t]
            # Build per-node infectious mask
            infectious = set(np.where(X==1)[0])
            # Accumulate candidate S-I infection events (unordered!)
            infect_events = []
            for (i,j) in this_contacts:
                # check S-I (i->j and j->i)
                if X[i]==0 and X[j]==1:
                    infect_events.append(i)
                elif X[j]==0 and X[i]==1:
                    infect_events.append(j)
            # Infection: unique S only, one attempt per contact per timestep
            if infect_events:
                infect_events = set(infect_events)  # de-duplicate: each S contacts any I at least once
                inf_trials = np.random.rand(len(infect_events))
                infected_this_step = [s for s, u in zip(infect_events, inf_trials) if u < beta]
                X[list(infected_this_step)] = 1
            # Recovery
            i_indices = np.where(X==1)[0]
            recovery_trials = np.random.rand(len(i_indices))
            recov_this_step = i_indices[recovery_trials < gamma]
            X[recov_this_step] = 2
        return dict(
            S_t=S_t, I_t=I_t, R_t=R_t,
            final_size=R_t[-1]/N,
            time_to_peak=time_to_peak,
            peak_I=peak_I
        )

    # 5. Run n_sim stochastic realizations and collect trajectories/outcomes
    all_S, all_I, all_R = [], [], []
    final_sizes, times_to_peak, peaks_I = [], [], []
    n_major_outbreaks = 0
    for sim in tqdm(range(n_sim)):
        result = one_sir_run(seed=12345+sim)
        all_S.append(result['S_t'])
        all_I.append(result['I_t'])
        all_R.append(result['R_t'])
        final_sizes.append(result['final_size'])
        times_to_peak.append(result['time_to_peak'])
        peaks_I.append(result['peak_I'])
        # Define 'major outbreak' as >1% infected at end
        if result['final_size'] > 0.01:
            n_major_outbreaks += 1
    major_outbreak_prob = n_major_outbreaks / n_sim
    mean_final_size = np.mean(final_sizes)
    sd_final_size = np.std(final_sizes)
    mean_time_to_peak = np.mean(times_to_peak)

    # 6. Save CSV: time-series mean ± 1 std for each compartment
    # Pad time series to same length
    maxlen = max(len(x) for x in all_S)
    def pad(xs):
        return [xi + [np.nan]*(maxlen-len(xi)) if len(xi)<maxlen else xi for xi in xs]
    arr_S = np.array(pad(all_S))
    arr_I = np.array(pad(all_I))
    arr_R = np.array(pad(all_R))
    data = pd.DataFrame({
        't': np.arange(maxlen),
        'S_mean': np.nanmean(arr_S,0)/N,
        'S_std': np.nanstd(arr_S,0)/N,
        'I_mean': np.nanmean(arr_I,0)/N,
        'I_std': np.nanstd(arr_I,0)/N,
        'R_mean': np.nanmean(arr_R,0)/N,
        'R_std': np.nanstd(arr_R,0)/N
    })
    data.to_csv(output_csv, index=False)
    # 7. Save plot (aggregate epi curves and histogram of final sizes and peak times)
    fig, axs = plt.subplots(1,3,figsize=(15,4))
    axs[0].plot(data['t'], data['I_mean'], label='I (mean)', color='red')
    axs[0].fill_between(data['t'], data['I_mean']-data['I_std'], data['I_mean']+data['I_std'], color='red', alpha=0.2)
    axs[0].plot(data['t'], data['S_mean'], label='S (mean)', color='blue')
    axs[0].plot(data['t'], data['R_mean'], label='R (mean)', color='green')
    axs[0].set_xlabel('Time step')
    axs[0].set_ylabel('Fraction of population')
    axs[0].set_title('SIR temporal activity-driven SIR')
    axs[0].legend()
    axs[1].hist(final_sizes, bins=30)
    axs[1].axvline(np.mean(final_sizes), color='k', linestyle='--', label='Mean')
    axs[1].set_title('Final epidemic size')
    axs[1].set_xlabel('R_final (fraction)')
    axs[2].hist(times_to_peak, bins=30)
    axs[2].set_title('Time to peak')
    axs[2].set_xlabel('Timesteps')
    axs[2].axvline(np.mean(times_to_peak), color='k', linestyle='--', label='Mean')
    plt.tight_layout()
    plt.savefig(output_png, dpi=120)
    plt.close()
    # Return major summary statistics
    return dict(
        major_outbreak_prob=major_outbreak_prob,
        mean_final_size=mean_final_size,
        sd_final_size=sd_final_size,
        mean_time_to_peak=mean_time_to_peak,
        n_major_outbreaks=n_major_outbreaks,
        runs=n_sim
    )

# PARAMETERS for scenario 0
data_dir = os.path.join(os.getcwd(),'output')
activity_path = os.path.join(data_dir,'exp3-powerlaw-activity.npy')
edge_path = os.path.join(data_dir,'exp3-temporal-contact-edges.csv')
N = 10000
Tmax = 2000
n_sim = 120  # Sufficient for robust statistics, can increase if time allows
beta = 1.55763
m = 2
gamma = 0.1

output_csv = os.path.join(os.getcwd(), 'output', 'results-01.csv')
output_png = os.path.join(os.getcwd(), 'output', 'results-01.png')

stats = run_temporal_activity_driven_sir(
    edge_path, activity_path, N, beta, gamma, m, Tmax, n_sim, output_csv, output_png
)