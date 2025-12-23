
# Reason: Implement scenario (i=1, j=2): Stochastic SEIR simulation on BA (Barabási–Albert) network with hub-based seeding
# Output: results-12.csv (.png)
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

np.random.seed(44) # Different reproducible seed for this scenario

# 1. Load BA network
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'ba-network.npz'))
N = G_csr.shape[0]

# 2. Define SEIR model schema (same as before)
SEIR_model_schema = (
    fg.ModelSchema("SEIR")
    .define_compartment(['S', 'E', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='progression', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='E', inducer='I', network_layer='contact_network_layer', rate='beta_edge')
)

params = dict(beta_edge=0.1378, sigma=0.2, gamma=0.43)
SEIR_instance = (
    fg.ModelConfiguration(SEIR_model_schema)
    .add_parameter(**params)
    .get_networks(contact_network_layer=G_csr)
)

# 3. Initial condition: seed 1 I and 3 E on highest-degree nodes
X0 = np.zeros(N, dtype=int)
degree_arr = np.array(G_csr.sum(axis=1)).flatten()
hub_indices = np.argsort(-degree_arr)[:4]  # Top 4 degrees in descending order
X0[hub_indices[0]] = 2  # 1 I
X0[hub_indices[1:]] = 1  # 3 E
initial_condition = {'exact': X0}

# 4. Simulation parameters
stop_condition = {'time': 120}
nsim = 100
variation_type = '90ci'

# 5. Run simulation
sim = fg.Simulation(SEIR_instance, initial_condition=initial_condition, stop_condition=stop_condition, nsim=nsim)
sim.run()
time, state_count, statecounts_lower_upper_bands = sim.get_results(variation_type=variation_type)

simulation_results = {'time': time}
for idx, state in enumerate(SEIR_model_schema.compartments):
    simulation_results[state] = state_count[idx, :]
    simulation_results[f"{state}_{variation_type}_lower"] = statecounts_lower_upper_bands[0, idx]
    simulation_results[f"{state}_{variation_type}_upper"] = statecounts_lower_upper_bands[1, idx]
data = pd.DataFrame(simulation_results)
csv_path = os.path.join(os.getcwd(), 'output', 'results-12.csv')
data.to_csv(csv_path, index=False)

# 6. Plot and save
png_path = os.path.join(os.getcwd(), 'output', 'results-12.png')
sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title="SEIR BA network, hub seeding", save_path=png_path)

hub_indices.tolist(), csv_path, png_path, params
