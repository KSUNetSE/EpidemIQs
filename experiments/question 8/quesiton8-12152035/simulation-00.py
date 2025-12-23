
# --- Step 1: Prepare scenario enumeration, parameter grid, and plan structure (i, j)
# Packages
import os
import numpy as np

# Model parameter lists provided:
beta_list = [0.15, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.3]
gamma_list = [0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15, 0.1]
xi_list = [0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.01]

n_scenarios = len(beta_list)
assert n_scenarios == len(gamma_list) == len(xi_list) == 13

# Decide on simulation settings—long enough time to reveal oscillations
time_max = 1500
n_realizations = 200  # Balanced for accuracy; can adjust if time too long
variation_type = '90ci'  # Main envelope for oscillation regime

# Filepaths (network already provided)
network_npz_path = '/Users/hosseinsamaei/phd/epidemiqs/output/watts-strogatz-graph.npz'

# Prescribed initial condition
initial_condition = {'percentage': {'U': 99, 'P': 1, 'F': 0}}  # 99% U, 1% P, 0% F

# Prepare outputs
task_setup = {
    'n_scenarios': n_scenarios,
    'beta_list': beta_list,
    'gamma_list': gamma_list,
    'xi_list': xi_list,
    'network_path': network_npz_path,
    'initial_condition': initial_condition,
    'variation_type': variation_type,
    'n_realizations': n_realizations,
    'time_max': time_max
}
task_setup

import fastgemf as fg
import os
# --- Step 4: For a single scenario (model 0), build ModelConfiguration, run a short test simulation, and verify output
beta_list = [0.15, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.3]
gamma_list = [0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15, 0.1]
xi_list = [0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.01]

from scipy import sparse
network_path = os.path.join(os.getcwd(), 'output', 'watts-strogatz-graph.npz')
G_csr = sparse.load_npz(network_path)

model_schema = (
    fg.ModelSchema("SIRS-Upf")
    .define_compartment(['U', 'P', 'F'])
    .add_network_layer('contact_network')
    .add_edge_interaction(
        name='trend_infection', from_state='U', to_state='P', inducer='P', network_layer='contact_network', rate='beta')
    .add_node_transition(
        name='fatigue', from_state='P', to_state='F', rate='gamma')
    .add_node_transition(
        name='forgetting', from_state='F', to_state='U', rate='xi')
)

# For scenario 0
beta, gamma, xi = beta_list[0], gamma_list[0], xi_list[0]
model_cfg = (
    fg.ModelConfiguration(model_schema)
    .add_parameter(beta=beta, gamma=gamma, xi=xi)
    .get_networks(contact_network=G_csr)
)

# Initial condition (percentage)
initial_condition = {'percentage': {'U': 99, 'P': 1, 'F': 0}}

# Stop condition and n realizations (short trial for speed)
sim = fg.Simulation(
    model_cfg,
    initial_condition=initial_condition,
    stop_condition={'time': 100},
    nsim=5
)
sim.run()
time, statecount, band = sim.get_results(variation_type='90ci')
# Pick a results path for temporary output
import pandas as pd
data = {'time': time}
for idx, comp in enumerate(model_schema.compartments):
    data[f'{comp}'] = statecount[idx]
    data[f'{comp}_90ci_lower'] = band[0][idx]
    data[f'{comp}_90ci_upper'] = band[1][idx]
df = pd.DataFrame(data)
results_path = os.path.join(os.getcwd(), 'output', 'results-00.csv')
df.to_csv(results_path, index=False)
plot_path = os.path.join(os.getcwd(), 'output', 'results-00.png')
sim.plot_results(time, statecount, variation_type='90ci', show_figure=False, save_figure=True, title=f'UPF trial: beta={beta}, gamma={gamma}, xi={xi}', save_path=plot_path)
results_path, plot_path
