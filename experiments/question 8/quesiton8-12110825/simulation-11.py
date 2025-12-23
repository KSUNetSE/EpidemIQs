
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

# Step 1: Load the Watts-Strogatz network
network_path_ws = '/Users/hosseinsamaei/phd/epidemiqs/output/network-wattsstrogatz.npz'
G_csr_ws = sparse.load_npz(network_path_ws)

# Step 2: Define SIR model schema (SIR, network edge infection, node recovery)
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network', rate='beta'
    )
    .add_node_transition(
        name='recovery', from_state='I', to_state='R', rate='gamma'
    )
)

# Step 3: Create model instance and configure network/params
sir_model_ws = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=0.2, gamma=1.0)
    .get_networks(contact_network=G_csr_ws)
)

# Step 4: Configure initial condition: one random infectious node, rest susceptible
initial_condition = {'percentage': {'I': 1, 'S': 99, 'R': 0}}
# This will seed 1% I (FastGEMF rounds for N=1000 to 1 random I, rest S)

# Step 5: Set globals for simulation
stop_time = 100  # Sufficient to cover wave and extinction in population 1000
nsim = 100  # Stochastic realizations for robust statistics
variation_type = '90ci'

# Step 6: Run simulation
sim_ws = fg.Simulation(sir_model_ws, initial_condition=initial_condition, stop_condition={'time': stop_time}, nsim=nsim)
sim_ws.run()

# Step 7: Gather/Save results
# (time, state_count, state_count_variation_bands)
time, state_count, statecounts_lower_upper_bands = sim_ws.get_results(variation_type=variation_type)

result_dict = {'time': time}
states = SIR_schema.compartments
state_count_mean = state_count
for i in range(len(states)):
    result_dict[f"{states[i]}"] = state_count_mean[i, :]/G_csr_ws.shape[0]  # fraction
    result_dict[f"{states[i]}_{variation_type}_lower"] = statecounts_lower_upper_bands[0, i] / G_csr_ws.shape[0]
    result_dict[f"{states[i]}_{variation_type}_upper"] = statecounts_lower_upper_bands[1, i] / G_csr_ws.shape[0]

out_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(out_dir, exist_ok=True)
res_csv = os.path.join(out_dir, 'results-11.csv')
pd.DataFrame(result_dict).to_csv(res_csv, index=False)

# Save figure of mean and CI bands
res_png = os.path.join(out_dir, 'results-11.png')
sim_ws.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title="WS-network SIR: R0=2, beta=0.2, gamma=1.0", save_path=res_png)

# Step 8: Extract/derive requested metrics: times to 10% and 50% I, and peak I (fraction)
I_frac = state_count[1] / G_csr_ws.shape[0]
t_10 = time[np.argmax(I_frac >= 0.10)] if np.any(I_frac >= 0.10) else np.nan
t_50 = time[np.argmax(I_frac >= 0.50)] if np.any(I_frac >= 0.50) else np.nan
I_peak = np.max(I_frac)

metrics = {'t_10': t_10, 't_50': t_50, 'I_peak': I_peak}
metrics_csv = os.path.join(out_dir, 'results-11-metrics.csv')
pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
