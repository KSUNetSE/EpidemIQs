
# Timeout workaround: For baseline, reduce to nsim=100 for feasibility, then upscale for main sweeps if runtime permits. Larger nsim will be used for threshold scenarios once baseline is validated for pipeline correctness.
# All other logic is unchanged from baseline scenario above.
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

# --- Load network
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

# Initial condition: all S except 1 random I
X0 = np.zeros(N, dtype=int)
seed = np.random.choice(N,10)
X0[seed] = 1
initial_condition = {'exact': X0}

nsim = 100  # Lower for feasibility in baseline validation
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 365.0}, nsim=nsim)
sim.run()

variation_type = "90ci"
result_outpath = os.path.join(os.getcwd(), 'output', 'results-xx.csv')
plot_outpath = os.path.join(os.getcwd(), 'output', 'results-xx.png')
time, state_count, statecounts_lower_upper_bands = sim.get_results(variation_type=variation_type)
compartments = SIR_model_schema.compartments
simulation_results = {'time': time}
for i in range(state_count.shape[0]):
    simulation_results[f"{compartments[i]}"] = state_count[i, :]
    simulation_results[f"{compartments[i]}_{variation_type}_lower"] = statecounts_lower_upper_bands[0,i]
    simulation_results[f"{compartments[i]}_{variation_type}_upper"] = statecounts_lower_upper_bands[1,i]
data = pd.DataFrame(simulation_results)
data.to_csv(result_outpath, index=False)

sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True, title="SIR Baseline (No Vaccination)", save_path=plot_outpath)

network_stats = {'nodes': N, 'edges': G_csr.nnz//2}
