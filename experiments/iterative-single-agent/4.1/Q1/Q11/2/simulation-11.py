
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os
# Run SEIR on ER (homogeneous mixing network)
N = 1000
network_path = os.path.join(os.getcwd(), 'output', 'network-er.npz')
G_csr = sparse.load_npz(network_path)
S, E, I, R = 'S', 'E', 'I', 'R'
SEIR_model = (
    fg.ModelSchema('SEIR')
    .define_compartment([S, E, I, R])
    .add_network_layer('contact')
    .add_edge_interaction('infection', from_state=S, to_state=E, inducer=I, network_layer='contact', rate='beta')
    .add_node_transition('latent', from_state=E, to_state=I, rate='sigma')
    .add_node_transition('recov', from_state=I, to_state=R, rate='gamma')
)
SEIR_instance = (
    fg.ModelConfiguration(SEIR_model)
    .add_parameter(beta=0.5, sigma=0.3333333333333333, gamma=0.2)
    .get_networks(contact=G_csr)
)
initial_condition = {'percentage': {'S': 99, 'E': 1, 'I': 0, 'R': 0}}
sim = fg.Simulation(SEIR_instance, initial_condition=initial_condition, stop_condition={'time': 180}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
for i in range(state_count.shape[0]):
    simulation_results[f'{SEIR_model.compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
'finished ER simulation'
# Chain of Thought:
# 1. Stochastic simulation using fastgemf (GEMFpy compatible):
#    - Network: BA (scale-free), previously saved
#    - Model: SEIR, compartments [S, E, I, R]
#    - Parameters: beta = 0.01642, sigma = 0.25, gamma = 0.1667
#    - Initial: 10 infected, 1990 susceptible.
#    - Save result to 'results-11.csv' and plot to 'results-11.png'
import fastgemf as fg
import os
from scipy import sparse
import numpy as np
import pandas as pd

S, E, I, R = 0, 1, 2, 3

# Load the BA network
network_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network-BA.npz'))

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
    .add_parameter(beta=0.01642, sigma=0.25, gamma=0.1667)
    .get_networks(contact=network_csr)
)

init_cond = {'percentage': {'S': 99.5, 'E': 0, 'I': 0.5, 'R': 0}}  # 10 infected, 1990 susceptible
sim = fg.Simulation(SEIR_instance, initial_condition=init_cond, stop_condition={'time': 180}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

time, state_count, *_ = sim.get_results()
results_dict = {'time': time}
for i, comp in enumerate(['S', 'E', 'I', 'R']):
    results_dict[comp] = state_count[i, :]
results_df = pd.DataFrame(results_dict)
results_df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
"Simulation completed"