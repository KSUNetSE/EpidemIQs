
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import pandas as pd

# 1. Load the SBM contact network
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)
N = 763

# 2. Define the SEICBR model schema
SEICBR_schema = (
    fg.ModelSchema("SEICBR")
    .define_compartment(['S', 'E', 'I', 'B', 'C', 'R'])
    .add_network_layer('contact')
    .add_edge_interaction(
        name='S_to_E', from_state='S', to_state='E', inducer='I',
        network_layer='contact', rate='beta')
    .add_node_transition(
        name='E_to_I', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(
        name='I_to_B', from_state='I', to_state='B', rate='gamma')
    .add_node_transition(
        name='B_to_C', from_state='B', to_state='C', rate='delta')
    .add_node_transition(
        name='C_to_R', from_state='C', to_state='R', rate='kappa')
)

# 3. Parameterize the model (from scenario/modeler)
params = {
    'beta': 0.1365451943519671,
    'sigma': 1.0,
    'gamma': 1.1111111111111112,
    'delta': 0.4,
    'kappa': 0.3333333333333333
}

SEICBR_instance = (
    fg.ModelConfiguration(SEICBR_schema)
    .add_parameter(**params)
    .get_networks(contact=G_csr)
)

# 4. Build the initial condition: 3 initial I in one house, all others S
house_map = pd.read_csv(os.path.join(os.getcwd(), 'output', 'house-mapping.csv'))
all_nodes_in_first_house = house_map[house_map['house'] == house_map['house'].unique()[0]].index.values
I_nodes = all_nodes_in_first_house[:3]
X0 = np.zeros(N, dtype=int)  # 0: S
X0[I_nodes] = 2  # I is index 2
initial_condition = {'exact': X0}

# 5. Set up simulation parameters
stop_time = 42  # Simulate 6 weeks: full epidemic
nsim = 500
variation_type = "90ci"

# 6. Run the simulation
sim = fg.Simulation(
    SEICBR_instance, 
    initial_condition=initial_condition,
    stop_condition={'time': stop_time},
    nsim=nsim
)
sim.run()

# 7. Save results and plot
compartments = ['S', 'E', 'I', 'B', 'C', 'R']
time, counts_mean, bounds = sim.get_results(variation_type=variation_type)
results = {'time': time}
for i, comp in enumerate(compartments):
    results[comp] = counts_mean[i]
    results[f"{comp}_{variation_type}_lower"] = bounds[0,i]
    results[f"{comp}_{variation_type}_upper"] = bounds[1,i]
data = pd.DataFrame(results)
output_csv = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data.to_csv(output_csv, index=False)
output_png = os.path.join(os.getcwd(), 'output', 'results-11.png')
sim.plot_results(time, counts_mean, variation_type=variation_type, show_figure=False, save_figure=True,
                 title="SEICBR SBM epidemic: SBM, beta=%.3f, nsim=%d" % (params['beta'], nsim), save_path=output_png)
