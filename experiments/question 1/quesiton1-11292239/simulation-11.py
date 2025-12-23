
# Reason: Implement scenario (i=1, j=1): Stochastic SEIR simulation on BA network with random seeding, using FastGEMF in CTMC mode
# Output: results-11.csv (.png)
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

np.random.seed(43) # Different seed for scenario B

# 1. Load BA network
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'ba-network.npz'))
N = G_csr.shape[0]

# 2. Define SEIR model schema (identical to ER)
SEIR_model_schema = (
    fg.ModelSchema("SEIR")
    .define_compartment(['S', 'E', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='progression', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='E', inducer='I', network_layer='contact_network_layer', rate='beta_edge')
)

# 3. Parameters for BA
params = dict(beta_edge=0.1378, sigma=0.2, gamma=0.43)
SEIR_instance = (
    fg.ModelConfiguration(SEIR_model_schema)
    .add_parameter(**params)
    .get_networks(contact_network_layer=G_csr)
)

# 4. Initial condition: 1 I, 3 E, remaining S (random)
X0 = np.zeros(N, dtype=int)
init_indices = np.random.choice(np.arange(N), size=4, replace=False)
X0[init_indices[0]] = 2  # 1 I
X0[init_indices[1:]] = 1  # 3 E
initial_condition = {'exact': X0}

# 5. Simulation parameters
stop_condition = {'time': 120}
nsim = 100
variation_type = '90ci'

# 6. Run simulation
sim = fg.Simulation(SEIR_instance, initial_condition=initial_condition, stop_condition=stop_condition, nsim=nsim)
sim.run()
time, state_count, statecounts_lower_upper_bands = sim.get_results(variation_type=variation_type)

simulation_results = {'time': time}
for idx, state in enumerate(SEIR_model_schema.compartments):
    simulation_results[state] = state_count[idx, :]
    simulation_results[f"{state}_{variation_type}_lower"] = statecounts_lower_upper_bands[0, idx]
    simulation_results[f"{state}_{variation_type}_upper"] = statecounts_lower_upper_bands[1, idx]
data = pd.DataFrame(simulation_results)
csv_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data.to_csv(csv_path, index=False)

# 7. Plot and save
png_path = os.path.join(os.getcwd(), 'output', 'results-11.png')
sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title="SEIR BA network, random seeding", save_path=png_path)

N, csv_path, png_path, init_indices.tolist(), params
