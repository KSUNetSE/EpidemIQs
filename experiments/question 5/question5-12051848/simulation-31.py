
# Scenario 3: Poisson(3) network, targeted vaccination (all degree=10 nodes removed, rare case; prediction is that effect is negligible)
import os
import numpy as np
import scipy.sparse as sparse
import fastgemf as fg
import pandas as pd

N = 10000
network_path = '/Users/hosseinsamaei/phd/epidemiqs/output/poisson3-z3-q4-network.npz'
stop_time = 60
nsim = 150
beta = 0.99
gamma = 1.0

np.random.seed(2026)
G_csr = sparse.load_npz(network_path)
degrees = np.array(G_csr.sum(axis=1)).flatten()
k10_idx = np.where(degrees == 10)[0]
node_states = np.zeros(N, dtype=int)
node_states[k10_idx] = 2  # R
nonvac_idx = np.setdiff1d(np.arange(N), k10_idx)
target_sus = np.where(node_states[nonvac_idx] == 0)[0]
n_I = max(1, int(np.floor(0.01 * len(nonvac_idx))))
I_sel = np.random.choice(nonvac_idx[target_sus], size=n_I, replace=False)
node_states[I_sel] = 1  # I
sir_schema = (
    fg.ModelSchema("SIR-Poisson-TargetedVac")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('net')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='net', rate='beta')
)
sir_instance = (
    fg.ModelConfiguration(sir_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(net=G_csr)
)
initial_condition = {'exact': node_states}
sim = fg.Simulation(sir_instance, initial_condition=initial_condition, stop_condition={'time': stop_time}, nsim=nsim)
sim.run()
time, state_mean, statebands = sim.get_results(variation_type="90ci")
sim.plot_results(time, state_mean, variation_type="90ci", show_figure=False, save_figure=True,
                 title="Poisson(3) Network SIR, All k=10 Vaccinated",
                 save_path=os.path.join(os.getcwd(), 'output', 'results-31.png'))
result = {'time': time}
for c_idx, comp in enumerate(['S', 'I', 'R']):
    result[comp] = state_mean[c_idx]
    result[f'{comp}_90ci_lower'] = statebands[0, c_idx]
    result[f'{comp}_90ci_upper'] = statebands[1, c_idx]
df = pd.DataFrame(result)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-31.csv'), index=False)
