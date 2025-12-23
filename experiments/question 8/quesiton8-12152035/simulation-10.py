
# Complete one more scan: scenario index 10 (tenth parameter set) for broad representativity
import fastgemf as fg
import os
from scipy import sparse
import pandas as pd
small_ws_path = os.path.join(os.getcwd(), 'output', 'watts-strogatz-small.npz')
G_csr = sparse.load_npz(small_ws_path)

beta_list = [0.15, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.3]
gamma_list = [0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15, 0.08, 0.08, 0.15, 0.15, 0.1]
xi_list = [0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.005, 0.015, 0.01]

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
scenario_index = 10
beta, gamma, xi = beta_list[scenario_index], gamma_list[scenario_index], xi_list[scenario_index]
model_cfg = (
    fg.ModelConfiguration(model_schema)
    .add_parameter(beta=beta, gamma=gamma, xi=xi)
    .get_networks(contact_network=G_csr)
)
initial_condition = {'percentage': {'U': 99, 'P': 1, 'F': 0}}
sim = fg.Simulation(
    model_cfg,
    initial_condition=initial_condition,
    stop_condition={'time': 120},  # very short test
    nsim=6
)
sim.run()
time, statecount, band = sim.get_results(variation_type='90ci')
data = {'time': time}
for idx, comp in enumerate(model_schema.compartments):
    data[f'{comp}'] = statecount[idx]
    data[f'{comp}_90ci_lower'] = band[0][idx]
    data[f'{comp}_90ci_upper'] = band[1][idx]
df = pd.DataFrame(data)
results_path = os.path.join(os.getcwd(), 'output', f'results-10.csv')
df.to_csv(results_path, index=False)
plot_path = os.path.join(os.getcwd(), 'output', f'results-10.png')
sim.plot_results(time, statecount, variation_type='90ci', show_figure=False, save_figure=True, title=f'UPF: beta={beta}, gamma={gamma}, xi={xi}', save_path=plot_path)
results_path, plot_path
