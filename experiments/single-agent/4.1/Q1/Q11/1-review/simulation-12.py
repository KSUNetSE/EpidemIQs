
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os
# Run SEIR on BA (heterogeneous network)
N = 1000
network_path = os.path.join(os.getcwd(), 'output', 'network-ba.npz')
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
    .add_parameter(beta=0.02886454529647018, sigma=0.3333333333333333, gamma=0.2)
    .get_networks(contact=G_csr)
)
initial_condition = {'percentage': {'S': 99, 'E': 1, 'I': 0, 'R': 0}}
sim = fg.Simulation(SEIR_instance, initial_condition=initial_condition, stop_condition={'time': 180}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-12.png'))
time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
for i in range(state_count.shape[0]):
    simulation_results[f'{SEIR_model.compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-12.csv'), index=False)
'finished BA simulation'