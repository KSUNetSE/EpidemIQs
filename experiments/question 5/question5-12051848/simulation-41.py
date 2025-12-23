
# Scenario 4: Tailored (2/3/10) network, random vaccination (sweep p in [0.7,0.72,0.74, ..., 0.8])
import os
import numpy as np
import scipy.sparse as sparse
import fastgemf as fg
import pandas as pd

N = 10000
network_path = '/Users/hosseinsamaei/phd/epidemiqs/output/tailored-z3-q4-network.npz'
stop_time = 60
nsim = 150
beta = 0.99
gamma = 1.0
seed_fraction = 0.01
variation_type = '90ci'
p_sweep = np.arange(0.7, 0.81, 0.02)  # 0.70, 0.72, ... 0.80
np.random.seed(2027)
G_csr = sparse.load_npz(network_path)

results_summary = []
for pi, p_vac in enumerate(p_sweep):
    n_vac = int(np.floor(p_vac * N))
    n_left = N - n_vac
    n_infected = max(1, int(np.floor(seed_fraction * n_left)))
    n_sus = n_left - n_infected
    node_states = np.zeros(N, dtype=int)
    vac_idx = np.random.choice(N, size=n_vac, replace=False)
    node_states[vac_idx] = 2  # R
    nonvac_idx = np.setdiff1d(np.arange(N), vac_idx)
    inf_idx = np.random.choice(nonvac_idx, size=n_infected, replace=False)
    node_states[inf_idx] = 1  # I
    # --- FastGEMF Model ---
    sir_schema = (
        fg.ModelSchema("SIR-Tailored-RandVac")
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
    time, state_mean, statebands = sim.get_results(variation_type=variation_type)
    sim.plot_results(time, state_mean, variation_type=variation_type, show_figure=False, save_figure=True,
                     title=f"Tailored network SIR {int(100*p_vac)}% random vac",
                     save_path=os.path.join(os.getcwd(), 'output', f'results-4{pi+1}.png'))
    result = {'time': time}
    for c_idx, comp in enumerate(['S', 'I', 'R']):
        result[comp] = state_mean[c_idx]
        result[f'{comp}_{variation_type}_lower'] = statebands[0, c_idx]
        result[f'{comp}_{variation_type}_upper'] = statebands[1, c_idx]
    df = pd.DataFrame(result)
    csv_path = os.path.join(os.getcwd(), 'output', f'results-4{pi+1}.csv')
    df.to_csv(csv_path, index=False)
    final_attack = float(state_mean[2, -1] - np.sum(node_states==2)) / N
    results_summary.append({'p_vac': p_vac, 'final_attack': final_attack, 'csv': csv_path, 'png': os.path.join(os.getcwd(), 'output', f'results-4{pi+1}.png')})
outdf = pd.DataFrame(results_summary)
outdf.to_csv(os.path.join(os.getcwd(), 'output', 'results-4attackrates.csv'), index=False)
