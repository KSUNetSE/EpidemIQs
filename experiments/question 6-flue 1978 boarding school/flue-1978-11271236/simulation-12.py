
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import os

# 1. Load network
network_path = os.path.join(os.getcwd(), 'output', 'network-complete.npz')
G_csr = sparse.load_npz(network_path)

# 2. Define ModelSchema
SEIBC_model_schema = (
    fg.ModelSchema("SEIBC")
    .define_compartment(['S', 'E', 'I', 'B', 'C'])
    .add_network_layer('contact_network_layer')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='E', inducer='I',
        network_layer='contact_network_layer', rate='beta')
    .add_node_transition(
        name='E_to_I', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(
        name='I_to_B', from_state='I', to_state='B', rate='kappa')
    .add_node_transition(
        name='B_to_C', from_state='B', to_state='C', rate='gamma')
)

# 3. Set Parameters for variation scenario (i=1, j=2): sigma = 0.5 (latent=2 days)
params = {'beta': 1.65, 'sigma': 0.5, 'kappa': 1.0, 'gamma': 0.3333}

SEIBC_instance = (
    fg.ModelConfiguration(SEIBC_model_schema)
    .add_parameter(**params)
    .get_networks(contact_network_layer=G_csr)
)

# 4. Initial condition (1 I, rest S)
N = G_csr.shape[0]
np.random.seed(12345)
X0 = np.zeros(N, dtype=int)
initial_I = np.random.choice(N, 1, replace=False)
X0[initial_I] = 2
initial_condition = {'exact': X0}

# 5. Simulate
nsim = 200
stop_days = 50
sim = fg.Simulation(SEIBC_instance, initial_condition=initial_condition,
                   stop_condition={'time': stop_days}, nsim=nsim)
sim.run()

# 6. Extract results (mean, 90% CI)
time, state_count, statecounts_lower_upper_bands = sim.get_results(variation_type="90ci")
compartments = ['S', 'E', 'I', 'B', 'C']
data_dict = {'time': time}
for idx, name in enumerate(compartments):
    data_dict[name] = state_count[idx, :]
    data_dict[f"{name}_90ci_lower"] = statecounts_lower_upper_bands[0, idx, :]
    data_dict[f"{name}_90ci_upper"] = statecounts_lower_upper_bands[1, idx, :]
data = pd.DataFrame(data_dict)

# 7. Save results
out_path = os.path.join(os.getcwd(), 'output', 'results-12.csv')
data.to_csv(out_path, index=False)

# 8. Save figure
sim.plot_results(time, state_count, variation_type="90ci", show_figure=False, save_figure=True,
                 title="SEIBC: sigma=0.5 (latent=2d), others baseline",
                 save_path=os.path.join(os.getcwd(), 'output', 'results-12.png'))

# 9. Summary stats: final attack rate, B peak
attack_rate = (data['C'].iloc[-1] + data['B'].iloc[-1]) / N
B_peak = data['B'].max()
B_peak_time = data['time'][data['B'].idxmax()]
