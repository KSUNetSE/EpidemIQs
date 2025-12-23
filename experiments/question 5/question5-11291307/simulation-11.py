
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

# Parameters
N = 10000
network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
G_csr = sparse.load_npz(network_path)

# Model schema
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contacts')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='mu')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contacts', rate='beta')
)

# Model config
SIR_instance = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=1.0, mu=1.0)
    .get_networks(contacts=G_csr)
)

np.random.seed(42)

# ---- Scenario 1: Random vaccination (v=0.75) ----
vac_frac = 0.75
num_vacc = int(N * vac_frac)
num_infect = 1
# indices to vaccinate, evenly at random
all_indices = np.arange(N)
vaccinated = np.random.choice(all_indices, num_vacc, replace=False)
not_vacc = np.setdiff1d(all_indices, vaccinated)
# seed an infectious
infected = np.random.choice(not_vacc, num_infect, replace=False)
# Build X0: 0=S, 1=I, 2=R
X0 = np.zeros(N, dtype=int)
X0[vaccinated] = 2
X0[infected] = 1
# check that exactly one I, 75% R, rest S
assert sum(X0==1) == 1
assert sum(X0==2) == num_vacc
initial_condition = {'exact': X0}

sim1 = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 40}, nsim=100)
sim1.run()
variation_type = "90ci"
time1, state_count1, band1 = sim1.get_results(variation_type=variation_type)
simulation_results1 = {'time': time1}
for i in range(state_count1.shape[0]):
    simulation_results1[f"{SIR_schema.compartments[i]}"] = state_count1[i, :]
    simulation_results1[f"{SIR_schema.compartments[i]}_{variation_type}_lower"] = band1[0, i]
    simulation_results1[f"{SIR_schema.compartments[i]}_{variation_type}_upper"] = band1[1, i]
data1 = pd.DataFrame(simulation_results1)
# Save results and plot
csv1 = os.path.join(os.getcwd(), 'output', 'results-11.csv')
png1 = os.path.join(os.getcwd(), 'output', 'results-11.png')
data1.to_csv(csv1, index=False)
sim1.plot_results(time1, state_count1, variation_type=variation_type, show_figure=False, save_figure=True, title="SIR Simulation: Random Vaccination v=0.75", save_path=png1)

r1_info = dict()
r1_info['csv'] = csv1
r1_info['png'] = png1
r1_info['summary'] = {
    'R_vac': np.sum(X0==2),
    'I_0': np.sum(X0==1),
    'S_0': np.sum(X0==0),
}

r1_info