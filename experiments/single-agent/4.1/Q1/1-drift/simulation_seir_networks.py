
# Chain of Thought for SEIR Simulation Setup:
# 1. Use FastGEMF to simulate SEIR over both ER and BA networks with N=2000, mean degree ~8.
# 2. Model Parameters, as above: beta=0.25, mu=1/3, gamma=1/5. This ensures R0 ~1.25 (homogeneous), >1 for both networks.
# 3. Initial condition: 1% infected, 1% exposed, rest susceptible. (Randomly assigned.)
# 4. Run for 200 days, 10 stochastic runs per network for variability.
# 5. Save outputs for each as results-ij.csv/png, where i=iteration, j=model (1:ER, 2:BA).

import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
import pandas as pd

# Load the networks
current_dir = os.getcwd()
ER_net = sparse.load_npz(os.path.join(current_dir, 'output', 'homogeneous_network.npz'))
BA_net = sparse.load_npz(os.path.join(current_dir, 'output', 'heterogeneous_network.npz'))

# SEIR model creation
SEIR_schema = (
    fg.ModelSchema("SEIR")
    .define_compartment(['S', 'E', 'I', 'R'])
    .add_network_layer('contacts')
    .add_node_transition(name='latent', from_state='E', to_state='I', rate='mu')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='E', inducer='I', network_layer='contacts', rate='beta')
)

param_dict = dict(beta=0.25, mu=1/3, gamma=1/5)

results_paths = []
plot_paths = []
model_names = ['Homogeneous (ER)', 'Heterogeneous (BA)']

for j, (net, net_name) in enumerate([(ER_net, 'ER'), (BA_net, 'BA')], start=1):
    SEIR_conf = (fg.ModelConfiguration(SEIR_schema)
                 .add_parameter(**param_dict)
                 .get_networks(contacts=net))
    N = net.shape[0]
    # Initial condition: 1% exposed, 1% infected, rest susceptible
    nE = max(1, int(N*0.01))
    nI = max(1, int(N*0.01))
    X0 = np.zeros(N, dtype=int)  # S=0, E=1, I=2, R=3
    X0[:nE] = 1
    X0[nE:nE+nI] = 2
    np.random.shuffle(X0)
    initial_condition = {'exact': X0}
    sim = fg.Simulation(SEIR_conf, initial_condition=initial_condition, stop_condition={'time': 200}, nsim=10)
    sim.run()
    # Save plot
    png_path = os.path.join(current_dir, 'output', f'results-1{j}.png')
    sim.plot_results(show_figure=False, save_figure=True, save_path=png_path)
    plot_paths.append(f'output/results-1{j}.png')
    # Save results as CSV
    time, state_count, *_ = sim.get_results()
    sim_results = {'time': time}
    for idx, comp in enumerate(SEIR_schema.compartments):
        sim_results[comp] = state_count[idx, :]
    df = pd.DataFrame(sim_results)
    csv_path = os.path.join(current_dir, 'output', f'results-1{j}.csv')
    df.to_csv(csv_path, index=False)
    results_paths.append(f'output/results-1{j}.csv')

(results_paths, plot_paths)