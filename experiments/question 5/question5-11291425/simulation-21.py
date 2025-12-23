
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import random
import pandas as pd
import os

# Scenario 2 (i=2, j=1): Random vaccination v=0.5 (50%)
np.random.seed(43)
random.seed(43)
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

# 50% vaccinated, from remainder 1% infected (of total N)
vacN = int(N * 0.5)
avail = np.arange(N)
vac_idx = np.random.choice(avail, vacN, replace=False)
remaining_idx = np.setdiff1d(avail, vac_idx)
n_I = max(1, int(0.01 * N))
I_idx = np.random.choice(remaining_idx, n_I, replace=False)
S_idx = np.setdiff1d(remaining_idx, I_idx)
# 0=S, 1=I, 2=R
X0 = np.zeros(N, dtype=int)
X0[vac_idx] = 2
X0[I_idx] = 1
# Safety: ensure sum correct
assert (np.sum(X0==0) + np.sum(X0==1) + np.sum(X0==2)) == N
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
data.to_csv(os.path.join(os.getcwd(), "output", "results-21.csv"), index=False)
sim.plot_results(time, statecount, variation_type=variation_type, show_figure=False, save_figure=True, title="Random Vaccination (v=0.5)", save_path=os.path.join(os.getcwd(), "output", "results-21.png"))
