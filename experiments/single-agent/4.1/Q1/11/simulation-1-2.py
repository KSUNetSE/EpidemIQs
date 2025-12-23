
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os

# Load heterogeneous (BA) network
G_ba = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_heterogeneous.npz'))

SEIR_model_schema = (
    fg.ModelSchema("SEIR")
    .define_compartment(['S', 'E', 'I', 'R'])
    .add_network_layer('contact_network')
    .add_edge_interaction('infection', from_state='S', to_state='E', inducer='I', network_layer='contact_network', rate='beta')
    .add_node_transition('latency', from_state='E', to_state='I', rate='sigma')
    .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
)

SEIR_instance_ba = (
    fg.ModelConfiguration(SEIR_model_schema)
    .add_parameter(beta=0.07, sigma=0.2, gamma=0.2)
    .get_networks(contact_network=G_ba)
)

# Use initial condition percentage ('S' + 'E' + 'I' + 'R' = 100)
init_cond = {'percentage': {'S': 99, 'E': 0, 'I': 1, 'R': 0}}  # 1% infected

sim = fg.Simulation(SEIR_instance_ba, initial_condition=init_cond, stop_condition={'time': 200}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-1-2.png'))

import pandas as pd
time, state_count, *_ = sim.get_results()
sim_data = {'time': time}
for i, name in enumerate(SEIR_model_schema.compartments):
    sim_data[name] = state_count[i, :]
pd.DataFrame(sim_data).to_csv(os.path.join(os.getcwd(), 'output', 'results-1-2.csv'), index=False)
