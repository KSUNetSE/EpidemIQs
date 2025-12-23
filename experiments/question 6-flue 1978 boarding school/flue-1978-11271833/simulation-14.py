
# Sensitivity: increase generation time (+0.5d); i.e., longer latent period (sigma down ~30%).
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import pandas as pd

network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)
N = 763

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

params_sigma_down = {
    'beta': 0.1365451943519671,
    'sigma': 0.7,  # increase latent period from 1.0 to ~1.43 days
    'gamma': 1.1111111111111112,
    'delta': 0.4,
    'kappa': 0.3333333333333333
}
SEICBR_instance = (
    fg.ModelConfiguration(SEICBR_schema)
    .add_parameter(**params_sigma_down)
    .get_networks(contact=G_csr)
)
house_map = pd.read_csv(os.path.join(os.getcwd(), 'output', 'house-mapping.csv'))
all_nodes_in_first_house = house_map[house_map['house'] == house_map['house'].unique()[0]].index.values
I_nodes = all_nodes_in_first_house[:3]
X0 = np.zeros(N, dtype=int)
X0[I_nodes] = 2
initial_condition = {'exact': X0}
stop_time = 42
nsim = 500
variation_type = "90ci"
sim = fg.Simulation(
    SEICBR_instance,
    initial_condition=initial_condition,
    stop_condition={'time': stop_time},
    nsim=nsim
)
sim.run()
compartments = ['S', 'E', 'I', 'B', 'C', 'R']
time, counts_mean, bounds = sim.get_results(variation_type=variation_type)
results = {'time': time}
for i, comp in enumerate(compartments):
    results[comp] = counts_mean[i]
    results[f"{comp}_{variation_type}_lower"] = bounds[0,i]
    results[f"{comp}_{variation_type}_upper"] = bounds[1,i]
data = pd.DataFrame(results)
output_csv = os.path.join(os.getcwd(), 'output', 'results-14.csv')
data.to_csv(output_csv, index=False)
output_png = os.path.join(os.getcwd(), 'output', 'results-14.png')
sim.plot_results(time, counts_mean, variation_type=variation_type, show_figure=False, save_figure=True,
                 title="SEICBR SBM: longer latent (sigma-0.7, Tgen+0.5d)", save_path=output_png)
