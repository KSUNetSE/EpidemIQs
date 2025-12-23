
import os, numpy as np, scipy.sparse as sparse, fastgemf as fg, pandas as pd

out_dir = os.path.join(os.getcwd(), 'output')

network_ER = sparse.load_npz(os.path.join(out_dir, 'network_ER.npz'))
network_BA = sparse.load_npz(os.path.join(out_dir, 'network_BA.npz'))
N = network_ER.shape[0]

sigma = 1/3
gamma = 1/4
beta_ER = 0.06226032726493215
beta_BA = 0.023778322624281367

SEIR_schema = (
    fg.ModelSchema('SEIR')
    .define_compartment(['S','E','I','R'])
    .add_network_layer('contact')
    .add_edge_interaction(name='infection', from_state='S', to_state='E', inducer='I', network_layer='contact', rate='beta')
    .add_node_transition(name='incubation', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
)

np.random.seed(42)
indices = np.random.permutation(N)
I_nodes = indices[:5]
E_nodes = indices[5:10]
X0 = np.zeros(N, dtype=int)
X0[E_nodes] = 1
X0[I_nodes] = 2
initial_condition = {'exact': X0}

results_paths = {}
for idx, (name, net, beta) in enumerate([('ER', network_ER, beta_ER), ('BA', network_BA, beta_BA)], start=1):
    config = (
        fg.ModelConfiguration(SEIR_schema)
        .add_parameter(beta=beta, sigma=sigma, gamma=gamma)
        .get_networks(contact=net)
    )
    sim = fg.Simulation(config, initial_condition=initial_condition, stop_condition={'time':200}, nsim=1)
    sim.run()
    # Save results
    time, state_counts, *_ = sim.get_results()
    data = {'time': time}
    for i, comp in enumerate(SEIR_schema.compartments):
        data[comp] = state_counts[i]
    df = pd.DataFrame(data)
    csv_path = os.path.join(out_dir, f'results-1{idx}.csv')
    df.to_csv(csv_path, index=False)
    results_paths[name] = csv_path
print(results_paths)