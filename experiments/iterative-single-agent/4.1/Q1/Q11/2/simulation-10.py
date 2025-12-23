
# Chain of Thought:
# 1. Stochastic simulation using fastgemf (GEMFpy compatible):
#    - Network: ER (homogeneous), previously saved
#    - Model: SEIR, compartments [S, E, I, R]
#    - Parameters: beta = 0.03455, sigma = 0.25, gamma = 0.1667
#    - Initial: 10 infected, 1990 susceptible.
#    - Save result to 'results-10.csv' and plot to 'results-10.png'
import fastgemf as fg
import os
from scipy import sparse
import numpy as np
import pandas as pd

S, E, I, R = 0, 1, 2, 3

# Load the ER network
network_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network-ER.npz'))

# Define SEIR model schema
SEIR_schema = (
    fg.ModelSchema('SEIR')
    .define_compartment(['S','E','I','R'])
    .add_network_layer('contact')
    .add_edge_interaction(name='inf', from_state='S', to_state='E', inducer='I', network_layer='contact', rate='beta')
    .add_node_transition(name='exposed', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(name='recover', from_state='I', to_state='R', rate='gamma')
)

SEIR_instance = (
    fg.ModelConfiguration(SEIR_schema)
    .add_parameter(beta=0.03455, sigma=0.25, gamma=0.1667)
    .get_networks(contact=network_csr)
)

init_cond = {'percentage': {'S': 99.5, 'E': 0, 'I': 0.5, 'R': 0}}  # 10 infected, 1990 susceptible
sim = fg.Simulation(SEIR_instance, initial_condition=init_cond, stop_condition={'time': 180}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-10.png'))

time, state_count, *_ = sim.get_results()
results_dict = {'time': time}
for i, comp in enumerate(['S', 'E', 'I', 'R']):
    results_dict[comp] = state_count[i, :]
results_df = pd.DataFrame(results_dict)
results_df.to_csv(os.path.join(os.getcwd(), 'output', 'results-10.csv'), index=False)
"Simulation completed"