
# Chain of Thought for Scenario 1: Poisson(3) Random Vaccination Sweep
# Goal: Simulate SIR dynamics on Poisson degree network (N=10,000, mean k=3) with random vaccination near p_c=0.75; sweep p in [0.7, 0.72, ..., 0.8] to check epidemic threshold.

import os
import numpy as np
import scipy.sparse as sparse
import fastgemf as fg
import pandas as pd
from tqdm import tqdm

# Parameters and settings
N = 10000
network_path = '/Users/hosseinsamaei/phd/epidemiqs/output/poisson3-z3-q4-network.npz'
stop_time = 60
nsim = 150
beta = 0.99  # Set for T ~ 1 (max percolation), per-step rate
gamma = 1.0
seed_fraction = 0.01  # 1% initially infected (post-vaccination; handeled in code as min 1)
variation_type = "90ci"
p_sweep = np.arange(0.7, 0.81, 0.02)  # fine sweep around threshold [0.70,0.72,...,0.80]
np.random.seed(2024)

# Load network
G_csr = sparse.load_npz(network_path)

results_summary = []
for pi, p_vac in enumerate(p_sweep):
    # 1. Assign compartments by random vaccination
    n_vac = int(np.floor(p_vac * N))
    n_left = N - n_vac
    n_infected = max(1, int(np.floor(seed_fraction * n_left)))
    n_sus = n_left - n_infected

    node_states = np.zeros(N, dtype=int)  # default all S=0
    # Randomly choose R (vaccinated)
    vac_idx = np.random.choice(N, size=n_vac, replace=False)
    node_states[vac_idx] = 2  # 2=R
    nonvac_idx = np.setdiff1d(np.arange(N), vac_idx)
    # From remaining nodes, randomly choose infected
    inf_idx = np.random.choice(nonvac_idx, size=n_infected, replace=False)
    node_states[inf_idx] = 1  # 1=I
    # All others remain S=0
    # Sanity check
    assert (np.sum(node_states == 2) == n_vac) and (np.sum(node_states == 1) == n_infected)
    # --- FastGEMF Model ---
    sir_schema = (
        fg.ModelSchema("SIR-Poisson3-RandVac")
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
    # Get results and save
    time, state_mean, statebands = sim.get_results(variation_type=variation_type)
    sim.plot_results(time, state_mean, variation_type=variation_type, show_figure=False, save_figure=True, title=f"Poisson(3) SIR with {int(100*p_vac)}% random vaccination", save_path=os.path.join(os.getcwd(), 'output', f'results-1{pi+1}.png'))
    # Save as csv
    result = {'time': time}
    for c_idx, comp in enumerate(['S', 'I', 'R']):
        result[comp] = state_mean[c_idx]
        result[f'{comp}_{variation_type}_lower'] = statebands[0, c_idx]
        result[f'{comp}_{variation_type}_upper'] = statebands[1, c_idx]
    df = pd.DataFrame(result)
    csv_path = os.path.join(os.getcwd(), 'output', f'results-1{pi+1}.csv')
    df.to_csv(csv_path, index=False)
    # Save final epidemic size stat for summary plot
    final_attack = float(state_mean[2, -1] - np.sum(node_states==2)) / N
    results_summary.append({'p_vac': p_vac, 'final_attack': final_attack, 'csv': csv_path, 'png': os.path.join(os.getcwd(), 'output', f'results-1{pi+1}.png')})

# Save epidemic size (attack rate) vs. vaccination threshold
outdf = pd.DataFrame(results_summary)
outdf.to_csv(os.path.join(os.getcwd(), 'output', 'results-1attackrates.csv'), index=False)
