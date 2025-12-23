
# Reasoning: Perform scenario 2 – SEIR on scale-free network.
# Nearly identical to scenario 1; only the network is changed (providing the scale-free .npz file).
# Use the same N, parameters, initialization logic, and output conventions.
import os
import numpy as np
import scipy.sparse as sparse
import fastgemf as fg
import pandas as pd

# 1. Load scale-free network
N = 10000
network_path = os.path.join(os.getcwd(), 'output', 'ScaleFree-Network.npz')
G_csr = sparse.load_npz(network_path)

# 2. Define SEIR model schema (same as before)
seir_schema = (
    fg.ModelSchema("SEIR")
    .define_compartment(['S', 'E', 'I', 'R'])
    .add_network_layer('contact_layer')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='E', inducer='I', network_layer='contact_layer', rate='beta')
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

# 4. Initial condition (10 random E, 1 random I, rest S): again use unique random assignments
rng = np.random.default_rng(43)  # different seed for robustness
all_nodes = np.arange(N)
exposed_nodes = rng.choice(all_nodes, size=10, replace=False)
remaining_nodes = np.setdiff1d(all_nodes, exposed_nodes)
infected_node = rng.choice(remaining_nodes, size=1, replace=False)

X0 = np.zeros(N, dtype=int)  # all susceptible initially
X0[exposed_nodes] = 1        # E=1
X0[infected_node] = 2        # I=2
initial_condition = {'exact': X0}

# 5. Simulation settings
nsim = 100
stop_condition = {'time': 200}
variation_type = '90ci'

# 6. Run simulation
sim = fg.Simulation(seir_instance, initial_condition=initial_condition, stop_condition=stop_condition, nsim=nsim)
sim.run()

# 7. Get results
results = {}
time, state_counts, statecounts_ci = sim.get_results(variation_type=variation_type)
results['time'] = time
comp_labels = ['S', 'E', 'I', 'R']
for i, label in enumerate(comp_labels):
    results[label] = state_counts[i]
    results[f'{label}_90ci_lower'] = statecounts_ci[0,i]
    results[f'{label}_90ci_upper'] = statecounts_ci[1,i]

# 8. Save results
out_csv = os.path.join(os.getcwd(), 'output', 'results-12.csv')
pd.DataFrame(results).to_csv(out_csv, index=False)

# 9. Save plot
out_png = os.path.join(os.getcwd(), 'output', 'results-12.png')
sim.plot_results(time, state_counts, variation_type=variation_type, show_figure=False, save_figure=True, title="SEIR on Scale-Free network", save_path=out_png)
