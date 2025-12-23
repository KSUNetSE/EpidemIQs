
import os
import fastgemf as fg
import scipy.sparse as sparse
import pandas as pd

# 1. Load the provided Watts–Strogatz network (CSR format)
network_path = '/Users/hosseinsamaei/phd/epidemiqs/output/ws-smallworld-1000-nodes.npz'
network_csr = sparse.load_npz(network_path)

# 2. Define the SIRS-UPF model schema
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

# 3. Configure the scenario-specific model instance
model_instance = (
    fg.ModelConfiguration(model_schema)
    .add_parameter(beta=0.8, gamma=0.3, xi=0.1)
    .get_networks(contact_network_layer=network_csr)
)

# 4. Specify initial condition as percentage (1% Posting, 99% Unaware, 0% Fatigued)
initial_condition = {'percentage': {'U': 99, 'P': 1, 'F': 0}}

# 5. Simulation setup
stop_time = 400  # days (long enough for oscillations to resolve)
nsim = 40  # number of realizations for confidence intervals
variation_type = '90ci'

sim = fg.Simulation(model_instance, initial_condition=initial_condition, stop_condition={'time': stop_time}, nsim=nsim)
sim.run()

time, state_count, statecounts_lower_upper_bands = sim.get_results(variation_type=variation_type)

# Prepare DataFrame for concise, analyzable result
simulation_results = {'time': time}
for i, c in enumerate(model_schema.compartments):
    simulation_results[f'{c}'] = state_count[i, :]
    simulation_results[f'{c}_{variation_type}_lower'] = statecounts_lower_upper_bands[0, i]
    simulation_results[f'{c}_{variation_type}_upper'] = statecounts_lower_upper_bands[1, i]
data = pd.DataFrame(simulation_results)

# Ensure output directory exists
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

# 6. Save the results and plot as requested
csv_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
fig_path = os.path.join(os.getcwd(), 'output', 'results-11.png')

data.to_csv(csv_path, index=False)
sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title='SIRS-UPF on small-world (baseline)', save_path=fig_path)
