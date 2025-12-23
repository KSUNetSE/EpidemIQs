
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
from collections import defaultdict

# Parameters
n = 1000
R0 = 3
recovery_rate = 0.1
alpha = 0.1
m = 2
# Calculate effective beta for the activity-driven network
k2 = m ** 2
k = m  # mean degree per snapshot, but for threshold use the mean excess degree q_time = (k2 - k) / k
q_time = (m ** 2 - m) / m
beta_temporal = R0 * recovery_rate / q_time

# Build temporal simulation engine (each snapshot is a new random network)
# Map node states: S=0, I=1, R=2

n_steps = 60
sim_steps = 10
results = defaultdict(list)
for sim in range(sim_steps):
    states = np.zeros(n, dtype=int)
    infected = np.random.choice(n, int(n * 0.01), replace=False)
    states[infected] = 1
    times = [0]
    S_hist = [np.sum(states == 0)]
    I_hist = [np.sum(states == 1)]
    R_hist = [np.sum(states == 2)]
    for t in range(1, n_steps+1):
        adj = np.zeros((n, n))
        for i in range(n):
            if np.random.rand() < alpha:
                partners = np.random.choice(np.delete(np.arange(n), i), m, replace=False)
                adj[i, partners] = 1
                adj[partners, i] = 1
        new_states = states.copy()
        # Infection
        for i in range(n):
            if states[i] == 0:
                infected_neighbors = np.where((adj[i] == 1) & (states == 1))[0]
                p_inf = 1 - (1 - beta_temporal) ** len(infected_neighbors)
                if np.random.rand() < p_inf:
                    new_states[i] = 1
        # Recovery
        for i in range(n):
            if states[i] == 1:
                if np.random.rand() < recovery_rate:
                    new_states[i] = 2
        states = new_states
        S_hist.append(np.sum(states == 0))
        I_hist.append(np.sum(states == 1))
        R_hist.append(np.sum(states == 2))
        times.append(t)
    results['S'].append(S_hist)
    results['I'].append(I_hist)
    results['R'].append(R_hist)
    results['time'] = times
# Average trajectories:
S_avg = np.mean(results['S'], axis=0)
I_avg = np.mean(results['I'], axis=0)
R_avg = np.mean(results['R'], axis=0)
import pandas as pd
df = pd.DataFrame({'time': times, 'S': S_avg, 'I': I_avg, 'R': R_avg})
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
'finished'
import fastgemf as fg
import scipy.sparse as sparse
import os

# Reload the static network CSR matrix
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_static.npz'))

# Define SIR schema
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery1',
        from_state='I',
        to_state='R',
        rate='gamma'
    )
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta'
    )
)

SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=0.0018239, gamma=0.2)
    .get_networks(contact_network_layer=G_csr)
)

import numpy as np
IC_vector = np.zeros(1000, dtype=int)
IC_vector[np.random.choice(1000, 5, replace=False)] = 1
initial_condition = {'exact': IC_vector}

sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 100}, nsim=1)
sim.run()
time, state_count, *_ = sim.get_results()
import pandas as pd
data = {'time': time}
for i, comp in enumerate(['S', 'I', 'R']):
    data[comp] = state_count[i, :]
pd.DataFrame(data).to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
'success'