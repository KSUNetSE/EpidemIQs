
# Scenario 1: Random vaccination sweep, starting slightly below threshold (v=0.6) through beyond threshold (v=0.9).
# We use 6 points (v: 0.6, 0.7, 0.75, 0.8, 0.85, 0.9).
# For each v, randomly assign v*N nodes as R, 1 seed as I, rest as S.
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

network_path = os.path.join('/Users/hosseinsamaei/phd/epidemiqs/output/network.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]

SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery',
        from_state='I',
        to_state='R',
        rate='gamma'
    )
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact_network_layer', rate='beta'
    )
)
SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
    .add_parameter(beta=1.0, gamma=1.0)
    .get_networks(contact_network_layer=G_csr)
)
vs = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
nsim = 100
# Iterate: each v mapped to i=(scenario index)
paths = []
plots = []
descriptions = []
for idx, v in enumerate(vs, start=1):
    X0 = np.zeros(N, dtype=int) # All S
    nR = int(round(v*N))
    nI = 1
    all_nodes = np.arange(N)
    R_nodes = np.random.choice(all_nodes, size=nR, replace=False)
    S_nodes = np.setdiff1d(all_nodes, R_nodes)
    I_node = np.random.choice(S_nodes, size=10)
    X0[R_nodes] = 2
    X0[I_node] = 1
    initial_condition = {'exact': X0}
    sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 365.0}, nsim=nsim)
    sim.run()
    variation_type = "90ci"
    result_outpath = os.path.join(os.getcwd(), 'output', f'results-{idx:02d}.csv')
    plot_outpath = os.path.join(os.getcwd(), 'output', f'results-{idx:02d}.png')
    time, state_count, statecounts_lower_upper_bands = sim.get_results(variation_type=variation_type)
    compartments = SIR_model_schema.compartments
    simulation_results = {'time': time}
    for i in range(state_count.shape[0]):
        simulation_results[f"{compartments[i]}"] = state_count[i, :]
        simulation_results[f"{compartments[i]}_{variation_type}_lower"] = statecounts_lower_upper_bands[0,i]
        simulation_results[f"{compartments[i]}_{variation_type}_upper"] = statecounts_lower_upper_bands[1,i]
    data = pd.DataFrame(simulation_results)
    data.to_csv(result_outpath, index=False)
    sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title=f"SIR Random Vaccination v={v:.2f}", save_path=plot_outpath)
    paths.append(result_outpath)
    plots.append((plot_outpath, N, G_csr.nnz//2))
    descriptions.append(f"Random vaccination at v={v:.2f}, {nR} nodes pre-immunized, one random I seed")
# Save meta
random_vacc_summary = {'cs_paths': paths, 'plot_paths': plots, 'desc': descriptions}
