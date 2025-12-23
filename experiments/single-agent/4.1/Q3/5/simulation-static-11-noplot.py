
import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
import pandas as pd

# Load static unweighted network
G_static_csr = sparse.load_npz(os.path.join('output', 'network-unweighted.npz'))

# Model parameters and initial conditions
N = 1000
mean_deg = 0.4
k2 = 0.546
R0 = 3
gamma = 0.2
q = (k2 - mean_deg) / mean_deg
beta = R0 * gamma / q

# Build Model
SIR_model_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact', rate='beta')
)

cfg = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact=G_static_csr)
)

init_cond = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

# Simulate, but just get results; skip plotting due to possible plotting issues
sim = fg.Simulation(cfg, initial_condition=init_cond, stop_condition={'time': 100}, nsim=10)
sim.run()
time, state_count, *_ = sim.get_results()
res1 = pd.DataFrame({'time': time, 'S': state_count[0], 'I': state_count[1], 'R': state_count[2]})
res1.to_csv(os.path.join('output','results-11.csv'), index=False)