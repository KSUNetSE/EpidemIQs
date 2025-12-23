
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import random
import pandas as pd
import os

# Retry Scenario 6 (i=6, j=1): Targeted vaccination (all degree-10 nodes, ~10.6%)
np.random.seed(47)
random.seed(47)
output_dir = os.path.join(os.getcwd(), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

network_path = '/Users/hosseinsamaei/phd/epidemiqs/output/configmodel-z3-q4.npz'
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]

sir_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I',
        network_layer='contact', rate='T'
    )
    .add_node_transition(
        name='removal', from_state='I', to_state='R', rate='gamma'
    )
)
sir_instance = (
    fg.ModelConfiguration(sir_schema)
    .add_parameter(T=1.0, gamma=1.0)
    .get_networks(contact=G_csr)
)

# The fix: Use .toarray().flatten() instead of .A1
if hasattr(G_csr, 'toarray'):
    degrees = G_csr.toarray().sum(axis=1).flatten()
else:
    degrees = np.array(G_csr.sum(axis=1)).flatten()
k10_idx = np.where(degrees == 10)[0]
remaining_idx = np.setdiff1d(np.arange(N), k10_idx)
n_I = max(1, int(0.01 * N))
I_idx = np.random.choice(remaining_idx, n_I, replace=False)
S_idx = np.setdiff1d(remaining_idx, I_idx)
X0 = np.zeros(N, dtype=int)
X0[k10_idx] = 2  # vaccinated
X0[I_idx] = 1
initial_condition = {'exact': X0}

sim = fg.Simulation(sir_instance, initial_condition=initial_condition, stop_condition={'time': 100}, nsim=200)
sim.run()
variation_type = "90ci"
time, statecount, bands = sim.get_results(variation_type=variation_type)
simulation_results = {"time": time}
for k in range(statecount.shape[0]):
    lab = sir_schema.compartments[k]
    simulation_results[f"{lab}"] = statecount[k, :]
    simulation_results[f"{lab}_{variation_type}_lower"] = bands[0, k, :]
    simulation_results[f"{lab}_{variation_type}_upper"] = bands[1, k, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), "output", "results-61.csv"), index=False)
sim.plot_results(time, statecount, variation_type=variation_type, show_figure=False, save_figure=True, title="Targeted Degree-10 Vaccination (~10.6%)", save_path=os.path.join(os.getcwd(), "output", "results-61.png"))
