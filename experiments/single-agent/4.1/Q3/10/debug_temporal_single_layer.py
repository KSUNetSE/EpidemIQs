
# Validate a single temporal layer loads correctly and is usable with SIR model to debug error
from scipy.sparse import load_npz
import fastgemf as fg
import os

# Try with just first temporal layer
A_t0 = load_npz(os.path.join(os.getcwd(), 'output', 'temporal_layer_0.npz'))

# Define SIR schema as above
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery1',
        from_state='I',
        to_state='R',
        rate='gamma'
    )
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta'
    )
)

SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=0.0018239, gamma=0.2)
    .get_networks(contact_network_layer=A_t0)
)

import numpy as np
IC_vector = np.zeros(1000, dtype=int)
IC_vector[np.random.choice(1000, 5, replace=False)] = 1
initial_condition = {'exact': IC_vector}

sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 100}, nsim=1)
sim.run()
time, state_count, *_ = sim.get_results()
time[:10]  # Just return first 10 times for debug