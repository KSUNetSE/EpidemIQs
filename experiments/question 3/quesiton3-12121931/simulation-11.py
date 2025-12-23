
# Reasoning: Start with scenario (i=1, j=1): SIR on static network using FastGEMF, as per plan.
# 1. Load the static network (CSR npz).
# 2. Define the SIR model in FastGEMF.
# 3. Parameterize with beta=7.5, gamma=1.0.
# 4. Initial state: 1 randomly infected node (rest S, none R).
# 5. 1000 stochastic realizations (nsim=1000).
# 6. Get results and plot using provided output path format.

import os
import numpy as np
import pandas as pd
import fastgemf as fg
import scipy.sparse as sparse
import random

# Parameters
N = 1000
beta = 7.5
gamma = 1.0
nsim = 1000
stop_time = 50  # Allow long enough for full epidemic fadeout.
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

# 1. Load network (CSR matrix)
network_path = '/Users/hosseinsamaei/phd/epidemiqs/output/network-static-n1000-k04.npz'
G_csr = sparse.load_npz(network_path)

# 2. Define SIR model schema
SIR_model_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contacts')
    .add_node_transition(
        name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I', network_layer='contacts', rate='beta')
)

# 3. Model parameterization
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contacts=G_csr)
)

# 4. Initial condition: single random infected, rest S
np.random.seed(42)  # fixed seed for reproducibility
random_infected = np.random.choice(N, 1)
X0 = np.zeros(N, dtype=int)  # All susceptible (state 0)
X0[random_infected[0]] = 1   # State 1 is 'I' in SIR
initial_condition = {'exact': X0}

# 5. Run simulation
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': stop_time}, nsim=nsim)
sim.run()
variation_type = "90ci"
time, state_count, statebands = sim.get_results(variation_type=variation_type)

# 6. Save results
data = {'time': time}
compartments = SIR_model_schema.compartments
for i, comp in enumerate(compartments):
    data[comp] = state_count[i, :]
    data[f'{comp}_90ci_lower'] = statebands[0, i, :]
    data[f'{comp}_90ci_upper'] = statebands[1, i, :]
results_df = pd.DataFrame(data)
results_path = os.path.join(output_dir, 'results-11.csv')
results_df.to_csv(results_path, index=False)
# 7. Plot
plot_path = os.path.join(output_dir, 'results-11.png')
sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title="SIR static network (N=1000, k=0.4, R0=3, FastGEMF)", save_path=plot_path)

# For output
caption_csv = 'SIR outbreak trajectories on static network (N=1000, k=0.4, 1000 runs; mean, 90% CI)'
caption_png = 'Epidemic curves Static SIR: mean, 90% CI – static network, R0=3, beta=7.5, gamma=1.0'

# Note: results are saved as per protocol.