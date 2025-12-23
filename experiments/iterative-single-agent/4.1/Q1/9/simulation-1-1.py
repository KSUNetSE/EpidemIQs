
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os

output_dir = os.path.join(os.getcwd(), 'output')
N = 1000

# Load networks
G_ER = sparse.load_npz(os.path.join(output_dir, 'network-ER.npz'))
G_BA = sparse.load_npz(os.path.join(output_dir, 'network-BA.npz'))

# Define model schema
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network', rate='beta')
)

# Settings from previous step
beta_ER = 0.04453478989510271
beta_BA = 0.02188068436453747
gamma = 0.14285714285714285

SIR_ER = fg.ModelConfiguration(SIR_schema).add_parameter(beta=beta_ER, gamma=gamma).get_networks(contact_network=G_ER)
SIR_BA = fg.ModelConfiguration(SIR_schema).add_parameter(beta=beta_BA, gamma=gamma).get_networks(contact_network=G_BA)

initial_cond = {'percentage': {'I': 1, 'S': 99, 'R': 0}}

# Run simulation for 120 days, 10 replicates each
sim_ER = fg.Simulation(SIR_ER, initial_condition=initial_cond, stop_condition={'time': 120}, nsim=10)
sim_BA = fg.Simulation(SIR_BA, initial_condition=initial_cond, stop_condition={'time': 120}, nsim=10)

sim_ER.run()
sim_BA.run()

# Plot and save figures
sim_ER.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(output_dir, 'results-1-ER.png'))
sim_BA.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(output_dir, 'results-1-BA.png'))

# Export data
import pandas as pd
time_ER, state_count_ER, *_ = sim_ER.get_results()
time_BA, state_count_BA, *_ = sim_BA.get_results()

results_ER = {'time': time_ER}
results_BA = {'time': time_BA}
for i, comp in enumerate(['S','I','R']):
    results_ER[comp] = state_count_ER[i, :]
    results_BA[comp] = state_count_BA[i, :]
df_ER = pd.DataFrame(results_ER)
df_BA = pd.DataFrame(results_BA)
df_ER.to_csv(os.path.join(output_dir, 'results-1-ER.csv'), index=False)
df_BA.to_csv(os.path.join(output_dir, 'results-1-BA.csv'), index=False)

# Record execution rationale
simu_reasoning = 'Simulations executed for SIR spread over two topologies (ER and BA, same avg degree), with the same initial infection (1%). Results recorded for epidemic curves and exported for comparative analysis.'
