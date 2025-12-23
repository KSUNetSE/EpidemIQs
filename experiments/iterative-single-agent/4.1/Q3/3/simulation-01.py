
# Chain of thought:
# The error happens when calling sim.plot_results in fastgemf. It could be due to None outputs or shape issues from the simulation.
# Let's (1) just run the simulation and (2) get the time and state_count output, and save those, to ensure the simulation runs and produces results.
# We'll omit plotting for now, and if data are OK, plot afterward.

import scipy.sparse as sparse
import os
import numpy as np
import fastgemf as fg
import pickle
import pandas as pd

network_path = os.path.join(os.getcwd(), 'output', 'network_static.npz')
G_csr = sparse.load_npz(network_path)
k1 = 181.464
k2 = 33135.578
n = 1000

# SIR param setup
gamma = 1/7 # recovery rate, per day
R0 = 3
beta = R0 * gamma / k1

# Initial conditions: 1% infected, randomly chosen, rest susceptible
num_infected = int(0.01 * n)
init_state = np.zeros(n, dtype=int)
init_state[:num_infected] = 1 # S=0, I=1, R=2
np.random.shuffle(init_state)

SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery1',
        from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta')
)

SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact_network_layer=G_csr)
)

static_X0 = {'exact': init_state}

sim = fg.Simulation(SIR_instance, initial_condition=static_X0, stop_condition={'time': 80}, nsim=10)
sim.run()
# Try: get results directly, skip plot for now.
time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i in range(state_count.shape[0]):
    simulation_results[f'{SIR_model_schema.compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-01.csv'), index=False)
return_vars = ['state_count','time']
