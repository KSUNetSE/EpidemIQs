
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import pandas as pd
import networkx as nx

# ER network: Same N, <k> as SBM
N = 763
mean_deg = 68.52
p_er = mean_deg/(N-1)
G_er = nx.erdos_renyi_graph(N, p_er, seed=42)
G_csr = nx.to_scipy_sparse_array(G_er, format='csr')

# Save this ER network for reference
sparse.save_npz(os.path.join(os.getcwd(), 'output', 'networker.npz'), G_csr)
# Rest as SBM: baseline params, initial 3 I in one house
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
# Seed: by default, use first 3 nodes as I (since there's no house structure)
X0 = np.zeros(N, dtype=int)
X0[:3] = 2  # I compartment
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
output_csv = os.path.join(os.getcwd(), 'output', 'results-16.csv')
data.to_csv(output_csv, index=False)
output_png = os.path.join(os.getcwd(), 'output', 'results-16.png')
sim.plot_results(time, counts_mean, variation_type=variation_type, show_figure=False, save_figure=True,
                 title="SEICBR ER network baseline", save_path=output_png)
