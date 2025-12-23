
# Plan: Setup and run FastGEMF simulation for SEIR on the two networks.
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import pandas as pd

output_dir = os.path.join(os.getcwd(), 'output')

# Define SEIR schema
SEIR_schema = (
    fg.ModelSchema("SEIR")
    .define_compartment(['S', 'E', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='exposed_to_infectious', from_state='E', to_state='I', rate='alpha')
    .add_node_transition(name='infect_to_recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='E', inducer='I',
        network_layer='contact_network_layer', rate='beta')
)

### Load networks
A_hom = sparse.load_npz(os.path.join(output_dir, 'network_homogeneous.npz'))
A_het = sparse.load_npz(os.path.join(output_dir, 'network_heterogeneous.npz'))

# Parameters
params_hom = {'beta': 0.3571428571428571, 'alpha': 0.3333333333333333, 'gamma': 0.14285714285714285}
params_het = {'beta': 0.017785820821887247, 'alpha': 0.3333333333333333, 'gamma': 0.14285714285714285}

# Initial condition: 10 infected, 990 susceptible
ic = {'percentage': {'S': 98, 'E': 0, 'I': 2, 'R': 0}}  # For n=1000

# Homogeneous network simulation
SEIRcfg_hom = (
    fg.ModelConfiguration(SEIR_schema)
    .add_parameter(**params_hom)
    .get_networks(contact_network_layer=A_hom)
)
sim_hom = fg.Simulation(SEIRcfg_hom, initial_condition=ic, stop_condition={'time':150}, nsim=10)
sim_hom.run()
sim_hom.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(output_dir,'results-1-1.png'))
# Save CSV
t_hom, states_hom, *_ = sim_hom.get_results()
df_hom = pd.DataFrame({'time': t_hom, 'S': states_hom[0], 'E': states_hom[1], 'I': states_hom[2], 'R': states_hom[3]})
df_hom.to_csv(os.path.join(output_dir,'results-1-1.csv'), index=False)

# Heterogeneous network simulation
SEIRcfg_het = (
    fg.ModelConfiguration(SEIR_schema)
    .add_parameter(**params_het)
    .get_networks(contact_network_layer=A_het)
)
sim_het = fg.Simulation(SEIRcfg_het, initial_condition=ic, stop_condition={'time':150}, nsim=10)
sim_het.run()
sim_het.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(output_dir,'results-1-2.png'))
t_het, states_het, *_ = sim_het.get_results()
df_het = pd.DataFrame({'time': t_het, 'S': states_het[0], 'E': states_het[1], 'I': states_het[2], 'R': states_het[3]})
df_het.to_csv(os.path.join(output_dir,'results-1-2.csv'), index=False)

# Return outputs
results_info = {
    'homogeneous': {'csv': os.path.join(output_dir,'results-1-1.csv'), 'png': os.path.join(output_dir,'results-1-1.png')},
    'heterogeneous': {'csv': os.path.join(output_dir,'results-1-2.csv'), 'png': os.path.join(output_dir,'results-1-2.png')}
}
results_info

import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os

# Load both networks
G_er = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network-er.npz'))
G_ba = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network-ba.npz'))

# Model setup: SEIR compartments
SEIR_schema = (
    fg.ModelSchema('SEIR')
    .define_compartment(['S', 'E', 'I', 'R'])
    .add_network_layer('contact')
    .add_node_transition('exposed_to_infectious', 'E', 'I', rate='sigma')
    .add_node_transition('infectious_to_removed', 'I', 'R', rate='gamma')
    .add_edge_interaction('infection', from_state='S', to_state='E', inducer='I', network_layer='contact', rate='beta')
)
# Parameters and IC
sigma = 1/3
gamma = 1/5
beta_er = 0.04115
beta_ba = 0.02238
N = G_er.shape[0]
# Start with 10 infectious, rest susceptible
init_cond = {'percentage': {'S': 98, 'E': 0, 'I': 1, 'R': 1}}  # Randomly assign 1% infected, 1% removed, none exposed
# Homogeneous (Erdos-Renyi)
SEIR_config_er = fg.ModelConfiguration(SEIR_schema).add_parameter(beta=beta_er, sigma=sigma, gamma=gamma).get_networks(contact=G_er)
# Heterogeneous (Barabasi-Albert)
SEIR_config_ba = fg.ModelConfiguration(SEIR_schema).add_parameter(beta=beta_ba, sigma=sigma, gamma=gamma).get_networks(contact=G_ba)
# Simulation
sim_er = fg.Simulation(SEIR_config_er, initial_condition=init_cond, stop_condition={'time': 180}, nsim=8)
sim_ba = fg.Simulation(SEIR_config_ba, initial_condition=init_cond, stop_condition={'time': 180}, nsim=8)
sim_er.run()
sim_ba.run()
# Save plots
sim_er.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-1-1.png'))
sim_ba.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-1-2.png'))
# Save csv
import pandas as pd
time, state_count, *_ = sim_er.get_results()
data = pd.DataFrame({'time': time, 'S': state_count[0], 'E': state_count[1], 'I': state_count[2], 'R': state_count[3]})
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-1-1.csv'), index=False)
time_ba, state_count_ba, *_ = sim_ba.get_results()
data_ba = pd.DataFrame({'time': time_ba, 'S': state_count_ba[0], 'E': state_count_ba[1], 'I': state_count_ba[2], 'R': state_count_ba[3]})
data_ba.to_csv(os.path.join(os.getcwd(), 'output', 'results-1-2.csv'), index=False)
# Return result file paths
res = {'results_paths': [os.path.join(os.getcwd(), 'output', 'results-1-1.csv'), os.path.join(os.getcwd(), 'output', 'results-1-2.csv')], 'fig_paths': [os.path.join(os.getcwd(), 'output', 'results-1-1.png'), os.path.join(os.getcwd(), 'output', 'results-1-2.png')]}
res