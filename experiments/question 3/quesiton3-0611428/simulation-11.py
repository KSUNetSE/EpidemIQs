
# Observe: Error likely means columns in the temporal-edge-events-csv are not named 'node1'/'node2'.
# Diagnose by inspecting available columns and a few rows.
import pandas as pd

path = '/Users/hosseinsamaei/phd/gemf_llm/output/temporal-edge-events-csv'
df = pd.read_csv(path)
columns = df.columns.tolist()
sample = df.head()
columns, sample
# Chain of Thought: The temporal event file columns are ['time','src','tgt'] not ['node1','node2'].
# Fix in the code: use (row.src, row.tgt) instead of (row.node1, row.node2) when building time_adjs.
# We will rerun the activity-driven simulation accordingly.

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

np.random.seed(1)

temp_event_path = '/Users/hosseinsamaei/phd/gemf_llm/output/temporal-edge-events-csv'
events = pd.read_csv(temp_event_path)
max_time = int(events['time'].max())
N = 1000
beta = 6.0
gamma = 1.0
T = max_time + 1
nsim = 100
P_trans = 1 - np.exp(-beta)
P_rec = 1 - np.exp(-gamma)

trajectories = []
t_end = np.zeros(nsim)
final_sizes = []

time_adjs = {t:[] for t in range(T)}
for row in events.itertuples(index=False):
    time_adjs[row.time].append((row.src, row.tgt))

for run in range(nsim):
    S = set(range(N))
    I = set([np.random.choice(list(S))])
    S = S - I
    R = set()
    sizes_S, sizes_I, sizes_R = [], [], []
    for t in range(T):
        #recover = set([i for i in I if np.random.rand() < P_rec])
        # = I - recover
        #R = R.union(recover)
        contacts = time_adjs[t]
        susceptible = S.copy()
        new_infected = set()
        for (u, v) in contacts:
            if u in I and v in susceptible and np.random.rand() < P_trans:
                new_infected.add(v)
            if v in I and u in susceptible and np.random.rand() < P_trans:
                new_infected.add(u)
        I = I.union(new_infected)
        recover = set([i for i in I if np.random.rand() < P_rec])
        I = I - recover
        R = R.union(recover)
        S = S - new_infected
        sizes_S.append(len(S))
        sizes_I.append(len(I))
        sizes_R.append(len(R))
        if not I:
            t_end[run] = t+1
            break
    else:
        t_end[run] = T
    trajectories.append(np.vstack([sizes_S, sizes_I, sizes_R]))
    final_sizes.append(len(R))

maxlen = int(np.max(t_end))
compartments = np.zeros((nsim, 3, maxlen), dtype=float)
for i, traj in enumerate(trajectories):
    T_seed = traj.shape[1]
    compartments[i, :, :T_seed] = traj
    if T_seed < maxlen:
        compartments[i,:,T_seed:] = traj[:,-1][:,None]

time_points = np.arange(maxlen)
summary = {
    'time': time_points,
    'S_mean': compartments[:,0,:].mean(axis=0),
    'I_mean': compartments[:,1,:].mean(axis=0),
    'R_mean': compartments[:,2,:].mean(axis=0),
    'S_std': compartments[:,0,:].std(axis=0),
    'I_std': compartments[:,1,:].std(axis=0),
    'R_std': compartments[:,2,:].std(axis=0),
    'I_p95': np.percentile(compartments[:,1,:], 95, axis=0),
    'I_p05': np.percentile(compartments[:,1,:], 5, axis=0),
}

output_csv = os.path.join(os.getcwd(), 'output', 'results-11.csv')
pd.DataFrame(summary).to_csv(output_csv, index=False)

plt.figure(figsize=(8,5))
plt.fill_between(time_points, summary['I_p05'], summary['I_p95'], color='orange', alpha=0.2, label='I 5-95% interval')
plt.plot(time_points, summary['I_mean'], label='I mean', color='red')
plt.plot(time_points, summary['S_mean'], label='S mean', color='blue')
plt.plot(time_points, summary['R_mean'], label='R mean', color='green')
plt.xlabel('Time (steps)')
plt.ylabel('Number of nodes')
plt.title('SIR on Activity-Driven Temporal Network')
plt.legend()
output_png = os.path.join(os.getcwd(), 'output', 'results-11.png')
plt.tight_layout()
plt.savefig(output_png)
plt.close()

final_sizes = np.array(final_sizes)
fs_df = pd.DataFrame({'final_size': final_sizes})
fs_df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11-finalsizes.csv'), index=False)

N_nodes = N
N_edges_per_t = np.mean([len(edges) for edges in time_adjs.values()])
{
    'N_nodes': N_nodes,
    'avg_contacts_per_timestep': N_edges_per_t,
    'final_size_mean': np.mean(final_sizes),
    'final_size_std': np.std(final_sizes)
}
# Chain of Thought:
# Careful review suggests:
# - Infection probability per contact should be handled accurately: For a single contact in CTMC SIR, the chance that S contacts I over dt=1 is (1 - exp(-beta)), which is correct for beta=6.0, but the process order and simultaneous updates might matter. All infectious nodes present at time t must be able to infect at time t, including just-infected ones if model allows.
# - States should be updated synchronously: Collect all new infections first (from S due to I), then update state arrays—not during iteration—which prevents state contamination mid-timestep.
# - Edge events at each timestep must be coded as undirected and all (u,v) must be considered both ways.
# - Make sure indexing of nodes and network (state S/I/R) is clean, with effective mapping and array logic for clarity.
# - Redo code with a clear synchronous update.
# - Plot and save as before for accurate diagnostics.

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

np.random.seed(1337)

temp_event_path = '/Users/hosseinsamaei/phd/gemf_llm/output/temporal-edge-events-csv'
events = pd.read_csv(temp_event_path)
max_time = int(events['time'].max())
N = 1000
beta = 6.0
gamma = 1.0
T = max_time + 1
nsim = 100

# Infection and recovery per time step, for any event across a single edge.
P_inf = 1 - np.exp(-beta)  # Per-contact, per timestep, from S to I
P_rec = 1 - np.exp(-gamma) # Per-timestep, I to R

# Pre-index temporal network by timestep for maximal speed; all events are undirected!
time_adjs = {t:[] for t in range(T)}
for row in events.itertuples(index=False):
    # Ensure undirected edge: store both (src, tgt) and (tgt, src)
    time_adjs[row.time].append( (row.src, row.tgt) )
    time_adjs[row.time].append( (row.tgt, row.src) )

trajectories = []
t_end = np.zeros(nsim)
final_sizes = []

for run in range(nsim):
    # 0=S, 1=I, 2=R
    state = np.zeros(N, dtype=int)
    patient_zero = np.random.choice(N)
    state[patient_zero] = 1

    sizes_S, sizes_I, sizes_R = [], [], []
    for t in range(T):
        # 1. Recoveries: All infectious nodes recover with probability P_rec this step
        recover_I = np.where(state == 1)[0]
        recovered = recover_I[np.random.rand(len(recover_I)) < P_rec]

        # 2. Transmission: Build infectious set before updates for synchronous logic
        infectious = np.where(state == 1)[0]
        susceptible = np.where(state == 0)[0]
        infected_this_step = set()
        contacts = time_adjs.get(t, [])
        for u, v in contacts:
            # Infectious u can infect susceptible v if event exists
            if (state[u] == 1) and (state[v] == 0):
                if np.random.rand() < P_inf:
                    infected_this_step.add(v)

        # 3. DO NOT infect or recover during event loop! Synchronous update below:
        next_state = state.copy()
        next_state[list(recovered)] = 2  # I->R
        next_state[list(infected_this_step)] = 1  # S->I
        state = next_state.copy()
        
        sizes_S.append(np.sum(state == 0))
        sizes_I.append(np.sum(state == 1))
        sizes_R.append(np.sum(state == 2))
        
        if np.sum(state == 1) == 0:
            t_end[run] = t+1
            break
    else:
        t_end[run] = T
    trajectories.append(np.vstack([sizes_S, sizes_I, sizes_R]))
    final_sizes.append(np.sum(state == 2))

maxlen = int(np.max(t_end))
compartments = np.zeros((nsim, 3, maxlen), dtype=float)
for i, traj in enumerate(trajectories):
    T_seed = traj.shape[1]
    compartments[i, :, :T_seed] = traj
    if T_seed < maxlen:
        compartments[i,:,T_seed:] = traj[:,-1][:,None]

time_points = np.arange(maxlen)
summary = {
    'time': time_points,
    'S_mean': compartments[:,0,:].mean(axis=0),
    'I_mean': compartments[:,1,:].mean(axis=0),
    'R_mean': compartments[:,2,:].mean(axis=0),
    'S_std':  compartments[:,0,:].std(axis=0),
    'I_std':  compartments[:,1,:].std(axis=0),
    'R_std':  compartments[:,2,:].std(axis=0),
    'I_p95': np.percentile(compartments[:,1,:], 95, axis=0),
    'I_p05': np.percentile(compartments[:,1,:], 5, axis=0),
}
output_csv = os.path.join(os.getcwd(), 'output', 'results-11.csv')
pd.DataFrame(summary).to_csv(output_csv, index=False)

# Plot
ymax = N+20
plt.figure(figsize=(8,5))
plt.fill_between(time_points, summary['I_p05'], summary['I_p95'], color='orange', alpha=0.2, label='I 5-95% interval')
plt.plot(time_points, summary['I_mean'], label='I mean', color='red')
plt.plot(time_points, summary['S_mean'], label='S mean', color='blue')
plt.plot(time_points, summary['R_mean'], label='R mean', color='green')
plt.xlabel('Time (steps)')
plt.ylabel('Number of nodes')
plt.title('REVISED: SIR on Activity-Driven Temporal Network')
plt.ylim([0, ymax])
plt.legend()
output_png = os.path.join(os.getcwd(), 'output', 'results-11.png')
plt.tight_layout()
plt.savefig(output_png)
plt.close()

final_sizes = np.array(final_sizes)
fs_df = pd.DataFrame({'final_size': final_sizes})
fs_df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11-finalsizes.csv'), index=False)

# Show mean and std of final epidemic sizes for diagnostics
{
    'final_size_mean': np.mean(final_sizes),
    'final_size_std': np.std(final_sizes)
}