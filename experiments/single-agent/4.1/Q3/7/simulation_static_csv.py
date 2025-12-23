
import fastgemf as fg
import scipy.sparse as sparse
import pandas as pd
import os

# 1. Load the aggregate static network
G_agg_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_aggregate.npz'))

# 2. SIR schema for static
SIR_model_schema = (
    fg.ModelSchema("SIR-Static")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network')
    .add_node_transition(
        name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network', rate='beta')
)

# 3. ModelConfiguration for static
beta_static = 0.0033113525454088783
params = {'beta': beta_static, 'gamma': 0.2}
static_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(**params)
    .get_networks(contact_network=G_agg_csr)
)

# 4. Initial condition (use percentage for random 1% infected)
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}

# 5. Simulate
sim_static = fg.Simulation(static_instance, initial_condition=initial_condition, stop_condition={'time': 80}, nsim=4)
sim_static.run()
# Try to save just CSV of state evolution
(time, state_count, *_ ) = sim_static.get_results()
sim_results = {}
sim_results['time'] = time
for i, c in enumerate(['S', 'I', 'R']):
    sim_results[c] = state_count[i,:]
df = pd.DataFrame(sim_results)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-10.csv'), index=False)
