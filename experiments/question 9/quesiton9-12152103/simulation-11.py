
# simulation-11.py: Periphery-seeded crisis scenario for SDL on core-periphery network
import os
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import fastgemf as fg

# Step 1: Load the network
netpath = '/Users/hosseinsamaei/phd/epidemiqs/output/network-core-periphery-banking.npz'
network_csr = sparse.load_npz(netpath)
N = network_csr.shape[0]
core_indices = np.arange(0, 20)
periphery_indices = np.arange(20, 100)

# Step 2: SDL schema and config (reuse from previous, e.g. simulation-10.py for identical params)
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

# Step 3: Initial condition, periphery-seeded (random periphery node)
np.random.seed(42)  # for reproducibility
periphery_seed = np.random.choice(periphery_indices)
X0 = np.zeros(N, dtype=int)
X0[periphery_seed] = 1  # only one random periphery node is D
initial_condition = {'exact': X0}

# Step 4: Simulation and output
sim = fg.Simulation(SDL_instance, initial_condition=initial_condition,
                    stop_condition={'time': 100.0}, nsim=1000)
sim.run()
variation_type = "90ci"
time, state_count, statecounts_lower_upper_bands = sim.get_results(variation_type=variation_type)

simulation_results = {'time': time}
for i, comp in enumerate(define_compartments):
    simulation_results[comp] = state_count[i, :]
    simulation_results[f"{comp}_{variation_type}_lower"] = statecounts_lower_upper_bands[0, i]
    simulation_results[f"{comp}_{variation_type}_upper"] = statecounts_lower_upper_bands[1, i]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), "output", "results-11.csv"), index=False)
sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True,
                 title="SDL periphery-seeded crisis (beta=0.3, gamma=0.5)",
                 save_path=os.path.join(os.getcwd(), "output", "results-11.png"))
