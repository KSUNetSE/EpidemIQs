
# Chain-of-Thought:
# The simulation crashed due to a plotting issue in fastgemf (possibly due to state variations or insufficient ensemble size, etc.).
# Let's re-run the simulation code, but this time:
# (A) Run only 1 simulation (to test if simulation itself or plotting is buggy)
# (B) Remove sim.plot_results, and only extract and save numerical results to CSV.

import os
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd

# Load static weighted network
net_path = os.path.join(os.getcwd(), 'output', 'network_static.npz')
G_csr = sparse.load_npz(net_path)
# Set up SIR model schema
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_layer')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_layer', rate='beta')
)
# Model configuration
SIR_instance = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=0.00182, gamma=0.2)
    .get_networks(contact_layer=G_csr)
)
# Initial condition: random percentage-based
init_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}
sim = fg.Simulation(SIR_instance, initial_condition=init_condition, stop_condition={'time': 150}, nsim=1)
sim.run()
time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i, comp in enumerate(SIR_schema.compartments):
    simulation_results[comp] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(),'output','results-1-1.csv'), index=False)
res = {'csv': os.path.join(os.getcwd(),'output','results-1-1.csv')}
res
