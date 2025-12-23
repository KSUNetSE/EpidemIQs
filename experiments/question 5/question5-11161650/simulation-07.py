
# A1 error fix: convert degrees extraction for sparse matrix to array
# .A1 not always supported; use .toarray().flatten() instead
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

network_path = os.path.join('/Users/hosseinsamaei/phd/epidemiqs/output/network.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]
degrees = np.array(G_csr.sum(axis=1)).flatten()
p10 = np.mean(degrees == 10)
deg10_nodes = np.where(degrees == 10)[0]
num_deg10 = len(deg10_nodes)

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

fs = [0.0, 0.945, 1.0]
nsim = 100
result_paths = []
plot_paths = []
descriptions = []
for j, f in enumerate(fs):
    X0 = np.zeros(N, dtype=int)
    nR = int(round(f * num_deg10))
    deg10_vacc = np.random.choice(deg10_nodes, size=nR, replace=False) if nR > 0 else []
    # Set all degree-10 vaccinates as R (2), others as S. Infect one not in R.
    if nR > 0:
        X0[deg10_vacc] = 2
    allowed_seeds = np.where(X0 != 2)[0]
    I_node = np.random.choice(allowed_seeds, size=1)
    X0[I_node] = 1
    initial_condition = {'exact': X0}
    sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 365.0}, nsim=nsim)
    sim.run()
    variation_type = "90ci"
    result_outpath = os.path.join(os.getcwd(), 'output', f'results-{j+7:02d}.csv')
    plot_outpath = os.path.join(os.getcwd(), 'output', f'results-{j+7:02d}.png')
    time, state_count, statecounts_lower_upper_bands = sim.get_results(variation_type=variation_type)
    compartments = SIR_model_schema.compartments
    simulation_results = {'time': time}
    for i in range(state_count.shape[0]):
        simulation_results[f"{compartments[i]}"] = state_count[i, :]
        simulation_results[f"{compartments[i]}_{variation_type}_lower"] = statecounts_lower_upper_bands[0,i]
        simulation_results[f"{compartments[i]}_{variation_type}_upper"] = statecounts_lower_upper_bands[1,i]
    data = pd.DataFrame(simulation_results)
    data.to_csv(result_outpath, index=False)
    sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title=f"SIR Degree-10 Targeted Vacc f={f:.3f}", save_path=plot_outpath)
    result_paths.append(result_outpath)
    plot_paths.append((plot_outpath, N, G_csr.nnz//2))
    total_covered = nR / N
    descriptions.append(f"Targeted degree-10 vaccination: fraction f={f:.3f} of degree-10 nodes ({nR}/{num_deg10}); population coverage {total_covered:.4f}; I seeded in S")
targeted_vacc_summary = {'csv_paths': result_paths, 'plot_paths': plot_paths, 'desc': descriptions}
