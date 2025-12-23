
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import random
import pandas as pd
import os

# Scenario 1 (i=1, j=1): Baseline (no vaccination)
np.random.seed(42)
random.seed(42)
output_dir = os.path.join(os.getcwd(), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the network
network_path = '/Users/hosseinsamaei/phd/epidemiqs/output/configmodel-z3-q4.npz'
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]

# SIR Model
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

# Parameters: T=1 (transmissibility), gamma=1 (recovery)
sir_instance = (
    fg.ModelConfiguration(sir_schema)
    .add_parameter(T=1.0, gamma=1.0)
    .get_networks(contact=G_csr)
)

# Baseline: 0% vaccinated, 1% infected, remainder susceptible.
n_S = int(N * 0.99)
n_I = N - n_S  # Ensure at least 1 infected
X0 = np.zeros(N, dtype=int)  # 0=S
infected_indices = np.random.choice(N, n_I, replace=False)
X0[infected_indices] = 1  # 1=I
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
data.to_csv(os.path.join(os.getcwd(), "output", "results-11.csv"), index=False)
sim.plot_results(time, statecount, variation_type=variation_type, show_figure=False, save_figure=True, title="SIR Baseline (No Vaccination)", save_path=os.path.join(os.getcwd(), "output", "results-11.png"))
