
# Reasoning for Iteration 4:
# Scenario 4: BA Network, baseline R0=3, beta_edge_ba=0.03003, gamma=0.1, initial 1 random I, nsim=100
# Save as results-13.csv/png.
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import os

np.random.seed(44)

# Model definition
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact', rate='beta')
)

# Load BA network
net_ba_path = '/Users/hosseinsamaei/phd/epidemiqs/output/barabasialbert-n1000-k10.npz'
G_ba = sparse.load_npz(net_ba_path)
N_ba = G_ba.shape[0]

# Model configuration
beta = 0.03003
gamma = 0.1
SIR_ba = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact=G_ba)
)

# IC: one random infectious
X0 = np.zeros(N_ba, dtype=int)
init_I_idx = np.random.choice(N_ba, 1, replace=False)
X0[init_I_idx] = 1
initial_condition = {'exact': X0}

nsim = 100
time_stop = 300
sim = fg.Simulation(SIR_ba, initial_condition=initial_condition, stop_condition={'time': time_stop}, nsim=nsim)
sim.run()

variation_type = "90ci"
time, state_count, statecounts_bands = sim.get_results(variation_type=variation_type)
# Save
outdir = os.path.join(os.getcwd(), 'output')
results_13_csv = os.path.join(outdir, 'results-13.csv')
sim_results = {'time': time}
for idx, comp in enumerate(SIR_schema.compartments):
    sim_results[f'{comp}'] = state_count[idx]
    sim_results[f'{comp}_{variation_type}_lower'] = statecounts_bands[0, idx]
    sim_results[f'{comp}_{variation_type}_upper'] = statecounts_bands[1, idx]
pd.DataFrame(sim_results).to_csv(results_13_csv, index=False)

# Plot
results_13_png = os.path.join(outdir, 'results-13.png')
sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title='BA network SIR, R0=3', save_path=results_13_png)
(results_13_csv, results_13_png)
