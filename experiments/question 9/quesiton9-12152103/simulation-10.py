
# -- Fix the error observed, numpy.ndarray does not have 'A1' attribute. Use flatten().
# simulation-10.py: Core-seeded crisis scenario (corrected degree calculation)
import os
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import fastgemf as fg

# Step 1: Load network
netpath = '/Users/hosseinsamaei/phd/epidemiqs/output/network-core-periphery-banking.npz'
network_csr = sparse.load_npz(netpath)

N = network_csr.shape[0]
core_indices = np.arange(0, 20)  # first 20 nodes are core
periphery_indices = np.arange(20, 100)

# Step 2: SDL model schema
define_compartments = ['S', 'D', 'L']
SDL_model_schema = (
    fg.ModelSchema("SDL")
    .define_compartment(define_compartments)
    .add_network_layer('contact_network_layer')
    .add_edge_interaction(
        name='distress_transmission', from_state='S', to_state='D', inducer='D',
        network_layer='contact_network_layer', rate='beta')
    .add_node_transition(
        name='liquidation', from_state='D', to_state='L', rate='gamma')
)

SDL_instance = (
    fg.ModelConfiguration(SDL_model_schema)
    .add_parameter(beta=0.3, gamma=0.5)
    .get_networks(contact_network_layer=network_csr)
)

# Step 4: Initial condition (core-hub as D)
degrees_core = network_csr[core_indices].sum(axis=1).flatten()  # shape: (20,)
core_hub_idx = core_indices[np.argmax(degrees_core)]
X0 = np.zeros(N, dtype=int)
X0[core_hub_idx] = 1  # only the hub is D
initial_condition = {'exact': X0}

# Step 5: Simulation settings
sim = fg.Simulation(SDL_instance, initial_condition=initial_condition,
                    stop_condition={'time': 100.0}, nsim=1000)
sim.run()
variation_type = "90ci"
time, state_count, statecounts_lower_upper_bands = sim.get_results(variation_type=variation_type)

# Results, save
simulation_results = {'time': time}
for i, comp in enumerate(define_compartments):
    simulation_results[comp] = state_count[i, :]
    simulation_results[f"{comp}_{variation_type}_lower"] = statecounts_lower_upper_bands[0, i]
    simulation_results[f"{comp}_{variation_type}_upper"] = statecounts_lower_upper_bands[1, i]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), "output", "results-10.csv"), index=False)
sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True,
                 title="SDL core-seeded crisis (beta=0.3, gamma=0.5)",
                 save_path=os.path.join(os.getcwd(), "output", "results-10.png"))
