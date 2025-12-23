
# The error is that .A1 is not available on the result of .sum(axis=1) if the output is already an np.ndarray.
# Instead, use np.array(...).flatten() to get the degree vector for the nodes.
# Fixing: deg = np.array(G_csr.sum(axis=1)).flatten() and rerun.
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import pandas as pd

N = 1000
beta = 7.5
gamma = 1.0
nsim = 1000
max_time = 365

network_static_path = os.path.join(os.getcwd(), 'output', 'activity-aggregated-network-corrected.npz')
G_csr = sparse.load_npz(network_static_path)

deg = np.array(G_csr.sum(axis=1)).flatten()
max_deg_node = np.argmax(deg)

SIR_model_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact')
    .add_edge_interaction(
        name='infection',
        from_state='S',
        to_state='I',
        inducer='I',
        network_layer='contact',
        rate='beta')
    .add_node_transition(
        name='recovery',
        from_state='I',
        to_state='R',
        rate='gamma')
)

SIR_config = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact=G_csr)
)

# Targeted initial condition: max-degree node
X0 = np.zeros(N, dtype=int)
X0[max_deg_node] = 1
initial_condition = {'exact': X0}

sim = fg.Simulation(SIR_config, initial_condition=initial_condition, stop_condition={'time': max_time}, nsim=nsim)
sim.run()
variation_type = '90ci'
time, state_count, statecounts_lower_upper_bands = sim.get_results(variation_type=variation_type)
result_csv_path = os.path.join(os.getcwd(), 'output', 'results-22.csv')
result_png_path = os.path.join(os.getcwd(), 'output', 'results-22.png')
simulation_results = {'time': time}
for i in range(state_count.shape[0]):
    simulation_results[f'SIR_{SIR_model_schema.compartments[i]}'] = state_count[i, :]
    simulation_results[f'SIR_{SIR_model_schema.compartments[i]}_90ci_lower'] = statecounts_lower_upper_bands[0, i]
    simulation_results[f'SIR_{SIR_model_schema.compartments[i]}_90ci_upper'] = statecounts_lower_upper_bands[1, i]
data = pd.DataFrame(simulation_results)
data.to_csv(result_csv_path, index=False)
sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title="SIR (beta=7.5, gamma=1) on aggregated static net, targeted max-degree I", save_path=result_png_path)
