
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os

# Load SIS network
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_sis.npz'))
# SIS Parameters
R0 = 2.5  # Compare to SIR
mean_k_sis = 10.1
mean_k2_sis = 112.694
gamma = 0.1  # average infectious period 10 days
beta = R0 * gamma * mean_k_sis / (mean_k2_sis - mean_k_sis)

# SIS Model Definition
SIS_model_schema = (
    fg.ModelSchema("SIS")
    .define_compartment(["S", "I"])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery', from_state='I', to_state='S', rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta')
)

SIS_instance = (
    fg.ModelConfiguration(SIS_model_schema)
    .add_parameter(beta=float(beta), gamma=gamma)
    .get_networks(contact_network_layer=G_csr)
)

N = G_csr.shape[0]
# Start with 5% infected, rest susceptible
s_init = int(0.95 * N)
i_init = N - s_init
initial_condition = {'percentage': {'S': int(100*s_init/N), 'I': int(100*i_init/N)}}

sim = fg.Simulation(SIS_instance, initial_condition=initial_condition, stop_condition={'time': 160}, nsim=6)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-12.png'))

import pandas as pd
time, state_count, *_ = sim.get_results()
data = pd.DataFrame({'time': time, 'S': state_count[0], 'I': state_count[1]})
data.to_csv(os.path.join(os.getcwd(),'output','results-12.csv'), index=False)

return_dict = {'beta': float(beta), 'gamma': gamma, 'R0': R0, 'init': initial_condition, 'network_path': os.path.join(os.getcwd(),'output','network_sis.npz'), 'result_path': os.path.join(os.getcwd(),'output','results-12.csv'), 'figure_path': os.path.join(os.getcwd(),'output','results-12.png')}
return_dict
