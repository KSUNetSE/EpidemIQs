
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import random

# Parameters
R0 = 4
mean_degree = 2.97  # from previous simulation
k2 = 11.806  # from previous simulation
N = 5000
beta = None
recover_rate = 1.0  # Set recovery rate to 1 per unit time (arbitrary, simplifies calculation)
q = (k2 - mean_degree) / mean_degree

beta = R0 * recover_rate / q

# Load the network
network_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_sim.npz'))

# Baseline, no vaccination
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery',
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
    .add_parameter(beta=beta, gamma=recover_rate)
    .get_networks(contact_network_layer=network_csr)
)

# Function to initialize population
N_inf_init = 30
init_cond = {'percentage': {'S': 100 - N_inf_init*100//N, 'I': N_inf_init*100//N, 'R': 0}}

# Simulate baseline epidemic (no vaccination) for reference
time_limit = 60
sim = fg.Simulation(SIR_instance, initial_condition=init_cond, stop_condition={'time': time_limit}, nsim=3)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-10.png'))
time, state_count, *_ = sim.get_results() # compartments: S=0, I=1, R=2
import pandas as pd
results = {'time': time}
results['S'] = state_count[0, :]
results['I'] = state_count[1, :]
results['R'] = state_count[2, :]
data = pd.DataFrame(results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-10.csv'), index=False)
data