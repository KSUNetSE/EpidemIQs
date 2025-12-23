
import os
import fastgemf as fg
import scipy.sparse as sparse
import pandas as pd

network_path = '/Users/hosseinsamaei/phd/epidemiqs/output/ws-smallworld-1000-nodes.npz'
network_csr = sparse.load_npz(network_path)

model_schema = (
    fg.ModelSchema('SIRS-UPF')
    .define_compartment(['U', 'P', 'F'])
    .add_network_layer('contact_network_layer')
    .add_edge_interaction(
        name='trend_spread', from_state='U', to_state='P', inducer='P', network_layer='contact_network_layer', rate='beta')
    .add_node_transition(
        name='fatigue', from_state='P', to_state='F', rate='gamma')
    .add_node_transition(
        name='forgetting', from_state='F', to_state='U', rate='xi')
)

model_instance = (
    fg.ModelConfiguration(model_schema)
    .add_parameter(beta=0.8, gamma=0.3, xi=0.2)  # FAST FORGETTING
    .get_networks(contact_network_layer=network_csr)
)
initial_condition = {'percentage': {'U': 99, 'P': 1, 'F': 0}}
stop_time = 400
nsim = 20  # reduce to 20 for runtime
variation_type = '90ci'

sim = fg.Simulation(model_instance, initial_condition=initial_condition, stop_condition={'time': stop_time}, nsim=nsim)
sim.run()

time, state_count, statecounts_lower_upper_bands = sim.get_results(variation_type=variation_type)
simulation_results = {'time': time}
for i, c in enumerate(model_schema.compartments):
    simulation_results[f'{c}'] = state_count[i, :]
    simulation_results[f'{c}_{variation_type}_lower'] = statecounts_lower_upper_bands[0, i]
    simulation_results[f'{c}_{variation_type}_upper'] = statecounts_lower_upper_bands[1, i]
data = pd.DataFrame(simulation_results)

output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(os.getcwd(), 'output', 'results-31.csv')
fig_path = os.path.join(os.getcwd(), 'output', 'results-31.png')
data.to_csv(csv_path, index=False)
sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title='SIRS-UPF, fast forgetting (xi=0.2)', save_path=fig_path)
