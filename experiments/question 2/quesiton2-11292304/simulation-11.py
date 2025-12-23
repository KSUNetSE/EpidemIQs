
# Reasoning for Iteration 2:
# Scenario 2: SIR on ER network, baseline R0=3, beta_edge_er=0.02981, gamma=0.1, N=1000, 1 random initial I, nsim=100
# Using FastGEMF, load provided ER network CSR file, construct exact X0 (one random I, rest S).
# ICs are exact (not random as percentage) to ensure match to task, and random seed is set for reproducibility.
# Save results and plot as required for scenario 2 (i=1, j=1: results-11.csv, results-11.png)
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# Model definition
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact', rate='beta')
)

# Load ER network from provided path
net_er_path = '/Users/hosseinsamaei/phd/epidemiqs/output/erdosrenyi-n1000-k10.npz'
G_er = sparse.load_npz(net_er_path)
N_er = G_er.shape[0]

# Model configuration
beta = 0.02981
gamma = 0.1
SIR_er = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact=G_er)
)

# Initial condition: one random I, rest S
X0 = np.zeros(N_er, dtype=int)  # All S by default
init_I_idx = np.random.choice(N_er, 1, replace=False)
X0[init_I_idx] = 1  # Set one as I
initial_condition = {'exact': X0}

# Simulation: nsim=100, stop at t=300
time_stop = 300
nsim = 100
sim = fg.Simulation(SIR_er, initial_condition=initial_condition, stop_condition={'time': time_stop}, nsim=nsim)
sim.run()

# Get results (90% CI for variation)
variation_type = "90ci"
time, state_count, statecounts_bands = sim.get_results(variation_type=variation_type)
# Save results (i=1, j=1)
outdir = os.path.join(os.getcwd(), 'output')
os.makedirs(outdir, exist_ok=True)
results_11_csv = os.path.join(outdir, 'results-11.csv')

sim_results = {'time': time}
for idx, comp in enumerate(SIR_schema.compartments):
    sim_results[f'{comp}'] = state_count[idx]
    sim_results[f'{comp}_{variation_type}_lower'] = statecounts_bands[0, idx]
    sim_results[f'{comp}_{variation_type}_upper'] = statecounts_bands[1, idx]
df = pd.DataFrame(sim_results)
df.to_csv(results_11_csv, index=False)

# Plot
results_11_png = os.path.join(outdir, 'results-11.png')
sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title='ER network SIR, R0=3', save_path=results_11_png)
(results_11_csv, results_11_png)
