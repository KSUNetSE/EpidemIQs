
# Reasoning for Iteration 3:
# Scenario 3: ER network, near-threshold, R0~1
# Use beta_edge = R0_target * gamma / <k> with <k>=10.064, gamma=0.1, so beta_edge=0.00994
# nsim=100, IC: 1 random I, rest S, save as results-12.csv/png.
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import os

np.random.seed(43)  # Different seed for new scenario

# Load ER network
net_er_path = '/Users/hosseinsamaei/phd/epidemiqs/output/erdosrenyi-n1000-k10.npz'
G_er = sparse.load_npz(net_er_path)
N_er = G_er.shape[0]

# Model definition
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact', rate='beta')
)

# Parameters for R0=1
beta = 0.00994
gamma = 0.1
SIR_er = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact=G_er)
)

# IC: one random infectious
X0 = np.zeros(N_er, dtype=int)
init_I_idx = np.random.choice(N_er, 1, replace=False)
X0[init_I_idx] = 1
initial_condition = {'exact': X0}
nsim = 100
time_stop = 300
sim = fg.Simulation(SIR_er, initial_condition=initial_condition, stop_condition={'time': time_stop}, nsim=nsim)
sim.run()

variation_type = "90ci"
time, state_count, statecounts_bands = sim.get_results(variation_type=variation_type)
# Save
outdir = os.path.join(os.getcwd(), 'output')
results_12_csv = os.path.join(outdir, 'results-12.csv')
sim_results = {'time': time}
for idx, comp in enumerate(SIR_schema.compartments):
    sim_results[f'{comp}'] = state_count[idx]
    sim_results[f'{comp}_{variation_type}_lower'] = statecounts_bands[0, idx]
    sim_results[f'{comp}_{variation_type}_upper'] = statecounts_bands[1, idx]
pd.DataFrame(sim_results).to_csv(results_12_csv, index=False)

# Plot
results_12_png = os.path.join(outdir, 'results-12.png')
sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title='ER SIR near threshold (R0~1)', save_path=results_12_png)
(results_12_csv, results_12_png)
