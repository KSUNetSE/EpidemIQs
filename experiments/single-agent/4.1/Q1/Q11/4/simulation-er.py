
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os

# ModelSchema for SEIR
SEIR_schema = (
    fg.ModelSchema("SEIR")
    .define_compartment(['S', 'E', 'I', 'R'])
    .add_network_layer('contacts')
    .add_node_transition(name='progression', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='E', inducer='I', network_layer='contacts', rate='beta')
)
# Load ER network
A_er = sparse.load_npz('output/network-er.npz')
config = (
    fg.ModelConfiguration(SEIR_schema)
    .add_parameter(beta=0.0631114, sigma=0.333333, gamma=0.2)
    .get_networks(contacts=A_er)
)
# Initial condition: 10 random E, rest S
N=2500
X0 = np.zeros(N, dtype=int)  # S=0
X0[np.random.choice(N, 10, replace=False)] = 1  # E=1
sim = fg.Simulation(config, initial_condition={'exact': X0}, stop_condition={'time':200}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-1-1.png'))
time, state_count, *_ = sim.get_results()
import pandas as pd
df = pd.DataFrame({'time': time, 'S': state_count[0], 'E': state_count[1], 'I': state_count[2], 'R': state_count[3]})
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-1-1.csv'), index=False)