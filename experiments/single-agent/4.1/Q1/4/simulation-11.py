
import fastgemf as fg
import numpy as np
import os
import pandas as pd
from scipy import sparse

# 1. Model schema
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(["S", "I", "R"])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network_layer', rate='beta')
)

# 2. Load network
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

# 3. Model config with parameters
sir_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=0.061012438703504375, gamma=0.2)
    .get_networks(contact_network_layer=G_csr)
)

# 4. Initial condition: corrected to sum to 100 per requirements
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}} # 1% infected, 99% susceptible

# 5. Simulation (up to 100 time units, 10 repetitions)
sim = fg.Simulation(sir_instance, initial_condition=initial_condition, stop_condition={'time': 100}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i, comp in enumerate(["S", "I", "R"]):
    simulation_results[comp] = state_count[i, :]
df = pd.DataFrame(simulation_results)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
import pandas as pd

# 1. Model Schema
schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_net')
    .add_edge_interaction('infection', from_state='S', to_state='I', inducer='I', network_layer='contact_net', rate='beta')
    .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
)

# 2. Load network
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

# 3. Model Configuration
params = {'beta': 0.03612045009495029, 'gamma': 0.142857}
config = (
    fg.ModelConfiguration(schema)
    .add_parameter(**params)
    .get_networks(contact_net=G_csr)
)

# 4. Initial condition (percentage)
init_cond = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

# 5. Simulate
sim = fg.Simulation(config, initial_condition=init_cond, stop_condition={'time': 120}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-11.png'))

# 6. Results
results = {}
time, state_count, *_ = sim.get_results()
results['time'] = time
for i, c in enumerate(schema.compartments):
    results[c] = state_count[i, :]
pd.DataFrame(results).to_csv(os.path.join(os.getcwd(),'output','results-11.csv'), index=False)

import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
# 1. Model schema
SIR_model_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network_layer', rate='beta')
)
# 2. Load the BA network
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))
# 3. Model configuration
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=0.012008119579258165, gamma=0.07)
    .get_networks(contact_network_layer=G_csr)
)
# 4. Initial condition: 99.7% S, 0.3% I, 0% R, rounded to nearest integer for a population of 1000
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}
# 5. Simulate, 1 realization
sim = fg.Simulation(
    SIR_instance,
    initial_condition=initial_condition,
    stop_condition={'time': 250},
    nsim=1
)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
# Save CSV
import pandas as pd
time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
for i in range(state_count.shape[0]):
    simulation_results[f'{SIR_model_schema.compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
