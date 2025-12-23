
# Reasoning (step 1, scenario 1)
# Baseline simulation for SEIR model on Erdős–Rényi (ER) network, per instructions.
# Steps:
# 1. Load ER network (N=10,000) from provided .npz path.
# 2. Create SEIR model schema in FastGEMF: S, E, I, R compartments.
# 3. Assign mechanistic transitions (S-(I)->E: beta=0.5, E->I: sigma=0.2, I->R: gamma=0.142857...)
# 4. Create parameterized ModelConfiguration; plug in loaded network for required layer.
# 5. Initialize nodes:
#         - 10 random, unique nodes as E (exposed)
#         - 1 random node as I (infectious)
#         - all others as S (susceptible)
#         - R=0
# 6. Setup simulation for nsim=100, 200 days, 90% CI for variability.
# 7. Save outputs to correct path.
# 8. Save output plot to required file.

import os
import numpy as np
import scipy.sparse as sparse
import fastgemf as fg
import pandas as pd

# 1. Load ER network (ensure output dir exists)
N = 10000
network_path = os.path.join(os.getcwd(), 'output', 'ER-Network.npz')
G_csr = sparse.load_npz(network_path)

# 2. Define SEIR model schema
seir_schema = (
    fg.ModelSchema("SEIR")
    .define_compartment(['S', 'E', 'I', 'R'])
    .add_network_layer('contact_layer')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='E', inducer='I',
        network_layer='contact_layer', rate='beta')
    .add_node_transition(
        name='incubation', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(
        name='recovery', from_state='I', to_state='R', rate='gamma')
)

# 3. Set parameter values
beta = 0.5
sigma = 0.2
gamma = 0.14285714285714285

seir_instance = (
    fg.ModelConfiguration(seir_schema)
    .add_parameter(beta=beta, sigma=sigma, gamma=gamma)
    .get_networks(contact_layer=G_csr)
)

# 4. Initial condition (10 random E, 1 random I, rest S): unique, non-overlapping
rng = np.random.default_rng(42)
all_nodes = np.arange(N)
exposed_nodes = rng.choice(all_nodes, size=10, replace=False)
remaining_nodes = np.setdiff1d(all_nodes, exposed_nodes)
infected_node = rng.choice(remaining_nodes, size=1, replace=False)

X0 = np.zeros(N, dtype=int)  # all susceptible
X0[exposed_nodes] = 1        # E=1
X0[infected_node] = 2        # I=2 (overrides E if overlapped, but with non-overlapping above guaranteed)

# 5. Prepare initial condition
initial_condition = {'exact': X0}

# 6. Setup simulation parameters
nsim = 100
stop_condition = {'time': 200}
variation_type = '90ci'

# 7. Run simulation
sim = fg.Simulation(seir_instance, initial_condition=initial_condition, stop_condition=stop_condition, nsim=nsim)
sim.run()

# 8. Get results
results = {}
time, state_counts, statecounts_ci = sim.get_results(variation_type=variation_type)
results['time'] = time
comp_labels = ['S', 'E', 'I', 'R']
for i, label in enumerate(comp_labels):
    results[label] = state_counts[i]
    results[f'{label}_90ci_lower'] = statecounts_ci[0,i]
    results[f'{label}_90ci_upper'] = statecounts_ci[1,i]

# 9. Save results
out_csv = os.path.join(os.getcwd(), 'output', 'results-11.csv')
pd.DataFrame(results).to_csv(out_csv, index=False)

# 10. Plot and save
out_png = os.path.join(os.getcwd(), 'output', 'results-11.png')
sim.plot_results(time, state_counts, variation_type=variation_type, show_figure=False, save_figure=True, title="SEIR on ER network", save_path=out_png)
