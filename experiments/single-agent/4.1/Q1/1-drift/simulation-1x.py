
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import pandas as pd

# Load networks
network_er = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_er.npz'))
network_ba = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_ba.npz'))
N = network_er.shape[0]

# SEIR Model schema
SEIR_schema = (
    fg.ModelSchema("SEIR")
    .define_compartment(['S', 'E', 'I', 'R'])
    .add_network_layer('contacts')
    .add_node_transition(
        name='exposed_to_infectious',
        from_state='E',
        to_state='I',
        rate='sigma')
    .add_node_transition(
        name='infectious_to_removed',
        from_state='I',
        to_state='R',
        rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='E', inducer='I',
        network_layer='contacts', rate='beta')
)

# Initialize (random configuration)
init_cond = {'percentage': {'S': 98, 'E': 0, 'I': 2, 'R': 0}}  # Random assignment

# Parameter sets for two networks
sigma = 0.33333  # 1/3 days

gamma = 0.16667  # 1/6 days
beta_er = 0.46667
beta_ba = 0.02022

# Simulation timespan
Tmax = 120
nsim = 10

# ER: Homogeneous
SEIR_cfg1 = (
    fg.ModelConfiguration(SEIR_schema).add_parameter(beta=beta_er, sigma=sigma, gamma=gamma)
    .get_networks(contacts=network_er)
)
sim1 = fg.Simulation(SEIR_cfg1, initial_condition=init_cond, stop_condition={'time': Tmax}, nsim=nsim)
sim1.run()
sim1.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
time, state_count, *_ = sim1.get_results()
res_dict1 = {'time': time}
for i, c in enumerate(SEIR_schema.compartments):
    res_dict1[c] = state_count[i, :]
pd.DataFrame(res_dict1).to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

# BA: Heterogeneous
SEIR_cfg2 = (
    fg.ModelConfiguration(SEIR_schema).add_parameter(beta=beta_ba, sigma=sigma, gamma=gamma)
    .get_networks(contacts=network_ba)
)
sim2 = fg.Simulation(SEIR_cfg2, initial_condition=init_cond, stop_condition={'time': Tmax}, nsim=nsim)
sim2.run()
sim2.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-12.png'))
time, state_count, *_ = sim2.get_results()
res_dict2 = {'time': time}
for i, c in enumerate(SEIR_schema.compartments):
    res_dict2[c] = state_count[i, :]
pd.DataFrame(res_dict2).to_csv(os.path.join(os.getcwd(), 'output', 'results-12.csv'), index=False)