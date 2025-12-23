
# Corrected logic for initial conditions: need to use percentage of nodes within 0-100, total sum = 100
# See if this fixes error, otherwise fallback to explicit 'exact' initialization
import fastgemf as fg
import numpy as np
from scipy import sparse
import os
import pandas as pd

N = 500
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

sir_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S','I','R'])
    .add_network_layer('er')
    .add_edge_interaction('infection', from_state='S', to_state='I', inducer='I', network_layer='er', rate='beta')
    .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
)
si_schema = (
    fg.ModelSchema('SI')
    .define_compartment(['S','I'])
    .add_network_layer('er')
    .add_edge_interaction('infection', from_state='S', to_state='I', inducer='I', network_layer='er', rate='beta')
)

R_0 = 2.3
mean_k = 25.34
mean_k2 = 664.508
gamma = 0.2
q = (mean_k2 - mean_k) / mean_k
beta = R_0 * gamma / q

sir_params = {'beta': beta, 'gamma': gamma}
si_params = {'beta': beta}

# The initial infected are 5 people out of 500 (1%), rest susceptible
sir_ic = {'percentage': {'S': 99, 'I': 1, 'R': 0}}
si_ic = {'percentage': {'S': 99, 'I': 1}}

sir_cfg = fg.ModelConfiguration(sir_schema).add_parameter(**sir_params).get_networks(er=G_csr)
sir_sim = fg.Simulation(sir_cfg, initial_condition=sir_ic, stop_condition={'time': 100}, nsim=5)
sir_sim.run()
sir_sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-1-1.png'))
t, C, *_ = sir_sim.get_results()
sir_df = pd.DataFrame({'time': t, 'S': C[0], 'I': C[1], 'R': C[2]})
sir_df.to_csv(os.path.join(os.getcwd(), 'output', 'results-1-1.csv'), index=False)

si_cfg = fg.ModelConfiguration(si_schema).add_parameter(**si_params).get_networks(er=G_csr)
si_sim = fg.Simulation(si_cfg, initial_condition=si_ic, stop_condition={'time': 100}, nsim=5)
si_sim.run()
si_sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-1-2.png'))
t2, C2, *_ = si_sim.get_results()
si_df = pd.DataFrame({'time': t2, 'S': C2[0], 'I': C2[1]})
si_df.to_csv(os.path.join(os.getcwd(), 'output', 'results-1-2.csv'), index=False)

(sir_params, si_params)