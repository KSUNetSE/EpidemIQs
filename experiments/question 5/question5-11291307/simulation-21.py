
import scipy.sparse as sparse
import numpy as np
import os
# Network info
N = 10000
adj = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))
deg = np.array(adj.sum(axis=1)).flatten()
deg10 = np.where(deg == 10)[0]
# Check degree-10 count for reasoning.
len_deg10 = len(deg10)
len_deg10
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

# Prepare scenario 2 (targeted vaccination, using k=10)
N = 10000
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))
deg = np.array(G_csr.sum(axis=1)).flatten()
deg10 = np.where(deg == 10)[0]  # correct new count: 1115
deg10_count = len(deg10)
not_deg10 = np.setdiff1d(np.arange(N), deg10)

np.random.seed(43)
# Infectious node among non-vaccinated
infected = np.random.choice(not_deg10, 1, replace=False)
X0 = np.zeros(N, dtype=int)
X0[deg10] = 2  # vaccinated
X0[infected] = 1  # infectious
# check
assert (X0==2).sum() == deg10_count
assert (X0==1).sum() == 1
initial_condition = {'exact': X0}

SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contacts')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='mu')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contacts', rate='beta')
)
SIR_instance = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=1.0, mu=1.0)
    .get_networks(contacts=G_csr)
)

sim2 = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 40}, nsim=100)
sim2.run()
variation_type = "90ci"
time2, state_count2, band2 = sim2.get_results(variation_type=variation_type)
simulation_results2 = {'time': time2}
for i in range(state_count2.shape[0]):
    simulation_results2[f"{SIR_schema.compartments[i]}"] = state_count2[i, :]
    simulation_results2[f"{SIR_schema.compartments[i]}_{variation_type}_lower"] = band2[0, i]
    simulation_results2[f"{SIR_schema.compartments[i]}_{variation_type}_upper"] = band2[1, i]
data2 = pd.DataFrame(simulation_results2)

csv2 = os.path.join(os.getcwd(), 'output', 'results-21.csv')
png2 = os.path.join(os.getcwd(), 'output', 'results-21.png')
data2.to_csv(csv2, index=False)
sim2.plot_results(time2, state_count2, variation_type=variation_type, show_figure=False, save_figure=True, title="SIR Simulation: Targeted Vaccination (deg=10)", save_path=png2)
r2_info = dict()
r2_info['csv'] = csv2
r2_info['png'] = png2
r2_info['summary'] = {'R_vac_deg10': deg10_count, 'I_0': 1, 'S_0': N - deg10_count - 1}
r2_info