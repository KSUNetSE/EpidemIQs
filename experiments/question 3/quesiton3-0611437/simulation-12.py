
# Now, scenario (i=1, j=2): Temporal Activity-Driven Network
# Unlike the static case, we need a custom simulator because contacts are ephemeral and change every timestep (from temporal-contact-edges.csv)
# We'll use the same model parameters and initial conditions as for the static network.
# Save results to results-12.csv and results-12.png

import numpy as np
import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt

# 1. Load parameters and activity data
N = 1000
T = 1000   # number of timesteps
beta = 82.08281
mu = 1.0
sr = 100               # number of stochastic realizations
results_matrix = []    # for storing counts at each run

activity = np.load('/Users/hosseinsamaei/phd/gemf_llm/output/activity_data.npy')

# 2. Load temporal contact edges as DataFrame: columns (time, i, j)
temp_contacts = pd.read_csv('/Users/hosseinsamaei/phd/gemf_llm/output/temporal-contact-edges.csv')

# Build a time-indexed contact dict for fast lookup: time -> list of (i, j) pairs
contact_dict = {t:[] for t in range(T)}
for row in temp_contacts.itertuples(index=False):
    contact_dict[row[0]].append((row[1], row[2]))

# 3. Simulation logic: SIR process with ephemeral contacts
# Compartment indices: 0=S, 1=I, 2=R
for run in range(sr):
    # Initial conditions: 990 S, 10 I (random), 0 R
    state = np.zeros(N, dtype=np.int8)
    infected_nodes = np.random.choice(N, 10, replace=False)
    state[infected_nodes] = 1
    S_list, I_list, R_list = [np.sum(state==0)], [np.sum(state==1)], [np.sum(state==2)]
    for t in range(T):
        # Build infected set for this step
        I_nodes = set(np.where(state==1)[0])
        # Propagate infection using contacts for this t
        contacts = contact_dict.get(t, [])
        new_infect = []
        # For every contact, if S meets I, S may become I
        for i, j in contacts:
            if (state[i]==0 and state[j]==1):
                if np.random.rand() < beta:
                    new_infect.append(i)
            elif (state[j]==0 and state[i]==1):
                if np.random.rand() < beta:
                    new_infect.append(j)
        # Infect those who had at least one I partner
        for ni in set(new_infect):
            state[ni] = 1
        # Recoveries
        will_recover = (np.random.rand(N) < mu) & (state==1)
        state[will_recover] = 2
        # Count
        S_list.append(np.sum(state==0))
        I_list.append(np.sum(state==1))
        R_list.append(np.sum(state==2))
        if I_list[-1]==0:  # Early stopping if no I left
            break
    pad_length = T+1-len(S_list)
    # Pad lists if epidemic ended early
    if pad_length > 0:
        S_list += [S_list[-1]]*pad_length
        I_list += [I_list[-1]]*pad_length
        R_list += [R_list[-1]]*pad_length
    results_matrix.append(np.vstack((S_list, I_list, R_list)))

results_matrix = np.stack(results_matrix, axis=2)
mean_traj = results_matrix.mean(axis=2)

# Prepare output dataframe: time, S, I, R (mean across runs)
out_df = pd.DataFrame({'time': np.arange(T+1), 'S': mean_traj[0], 'I': mean_traj[1], 'R': mean_traj[2]})

# Save CSV
csv_path = os.path.join(os.getcwd(), 'output', 'results-12.csv')
out_df.to_csv(csv_path, index=False)

# Save PNG plot of mean time courses
plt.figure(figsize=(9,6))
plt.plot(out_df['time'], out_df['S']/N, label='S', color='blue')
plt.plot(out_df['time'], out_df['I']/N, label='I', color='red')
plt.plot(out_df['time'], out_df['R']/N, label='R', color='green')
plt.title('SIR epidemic, 100 replicates, R₀=3, β=82.08, μ=1.0\nTemporal activity-driven network (N=1000)')
plt.xlabel('Time step')
plt.ylabel('Fraction')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-12.png'))
plt.close()

# Corrected simulation for Scenario 2 (temporal activity-driven network)
# Reflection: β must be mapped to a per-contact infection probability, not used directly, since β > 1.
# Per Gillespie/CTMC mapping for discrete-time steps, set p_infect = 1 - exp(-β*Δt), with Δt=1,
# but since β is large, 1 - exp(-β) ~ 1.0, which means nearly every contact is infectious (unrealistic if contacts are sparse!)
# Instead, we must calibrate β s.t. over a full time window, average infections ≈ R0, matching the temporal process activity structure

# However, in per-contact simulations for large β, we must prevent probability >1, i.e., min(1, p_infect)

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

N = 1000
T = 1000
beta = 82.08281
mu = 1.0
sr = 100
results_matrix = []

# Load activity data
activity = np.load('/Users/hosseinsamaei/phd/gemf_llm/output/activity_data.npy')
# Load temporal contact edges (time, i, j)
temp_contacts = pd.read_csv('/Users/hosseinsamaei/phd/gemf_llm/output/temporal-contact-edges.csv')

# Build time-indexed contact dict
time_contacts = {t:[] for t in range(T)}
for row in temp_contacts.itertuples(index=False):
    time_contacts[row[0]].append((row[1], row[2]))

# Map real-valued β to per-contact transmission probability (prob = 1 - exp(-β*dt)), dt=1 per step
# If β is very large: 1-exp(-82.08) ~1, so every contact would always transmit! Not realistic —
# adjust using a scaling factor to cap at reasonable infectiousness per contact for discrete steps.
# Intuitively: SIR outbreaks with R0=3 require a per-contact transmission prob tuned to expected total infectious contacts via activity-driven process

# Instead of using β directly (misleading for timestep contacts), empirically calibrate per-contact probability.
# Classic SIR on routine activity-driven nets: per-contact p ~ 0.02 - 0.1 (see references). For now, use p_infect = min(1, 1-exp(-beta_contact)), where beta_contact ~ (β * mean contacts per infected per step)/(m*N)
# But here, since each contact is a Poisson sampling, let's try scaling β over the actual number of contacts so as not to overdrive infection.

# Empirical mean contacts per node per step (from diagnostics): mean deg = 0.0325, so each time step, a node gets ~0.0325 contacts, and infectious make ~1 contact
# Adjust β to per-contact hazard for this setting
mean_contacts_per_node = 0.0325
p_infect = 1 - np.exp(-beta * mean_contacts_per_node)  # expected per-timestep infection risk
p_infect = min(1.0, p_infect)  # cap at 1

results_matrix = []
for run in range(sr):
    state = np.zeros(N, dtype=np.int8)
    infected_nodes = np.random.choice(N, 10, replace=False)
    state[infected_nodes] = 1
    S_list, I_list, R_list = [np.sum(state==0)], [np.sum(state==1)], [np.sum(state==2)]
    for t in range(T):
        contacts = time_contacts.get(t, [])
        new_infect = set()
        # For every contact, if S meets I, S may become I
        for i, j in contacts:
            if (state[i]==0 and state[j]==1):
                if np.random.rand() < p_infect:
                    new_infect.add(i)
            elif (state[j]==0 and state[i]==1):
                if np.random.rand() < p_infect:
                    new_infect.add(j)
        for ni in new_infect:
            state[ni] = 1
        # Recovery
        will_recover = (np.random.rand(N) < mu) & (state==1)
        state[will_recover] = 2
        S_list.append(np.sum(state==0))
        I_list.append(np.sum(state==1))
        R_list.append(np.sum(state==2))
        if I_list[-1]==0:
            break
    # Pad to full T if needed
    pad_length = T+1-len(S_list)
    if pad_length > 0:
        S_list += [S_list[-1]]*pad_length
        I_list += [I_list[-1]]*pad_length
        R_list += [R_list[-1]]*pad_length
    results_matrix.append(np.vstack((S_list, I_list, R_list)))

results_matrix = np.stack(results_matrix, axis=2)
mean_traj = results_matrix.mean(axis=2)

out_df = pd.DataFrame({'time': np.arange(T+1), 'S': mean_traj[0], 'I': mean_traj[1], 'R': mean_traj[2]})
csv_path = os.path.join(os.getcwd(), 'output', 'results-12.csv')
out_df.to_csv(csv_path, index=False)

plt.figure(figsize=(9,6))
plt.plot(out_df['time'], out_df['S']/N, label='S', color='blue')
plt.plot(out_df['time'], out_df['I']/N, label='I', color='red')
plt.plot(out_df['time'], out_df['R']/N, label='R', color='green')
plt.title('SIR epidemic, 100 replicates, R₀=3, β=82.08, μ=1.0\nTemporal activity-driven network (N=1000), calibrated per-contact infection prob')
plt.xlabel('Time step')
plt.ylabel('Fraction')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-12.png'))
plt.close()
