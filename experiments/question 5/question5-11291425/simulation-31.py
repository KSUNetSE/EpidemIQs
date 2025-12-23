
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import random
import pandas as pd
import os

# Scenario 3 (i=3, j=1): Random vaccination v=0.7 (70%)
np.random.seed(44)
random.seed(44)
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

# 70% vaccinated, from remainder 1% infected (of total N)
vacN = int(N * 0.7)
avail = np.arange(N)
vac_idx = np.random.choice(avail, vacN, replace=False)
remaining_idx = np.setdiff1d(avail, vac_idx)
n_I = max(1, int(0.01 * N))
I_idx = np.random.choice(remaining_idx, n_I, replace=False)
S_idx = np.setdiff1d(remaining_idx, I_idx)
X0 = np.zeros(N, dtype=int)
X0[vac_idx] = 2
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
data.to_csv(os.path.join(os.getcwd(), "output", "results-31.csv"), index=False)
sim.plot_results(time, statecount, variation_type=variation_type, show_figure=False, save_figure=True, title="Random Vaccination (v=0.7)", save_path=os.path.join(os.getcwd(), "output", "results-31.png"))
