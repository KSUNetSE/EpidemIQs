
# Chain of Thought: Previous code fails because the file '/Users/hosseinsamaei/phd/gemf_llm/output/agg-static-network-npz' cannot be found.
# Diagnostic: Check current working directory, list the contents of 'output', and look for static network files or variants (maybe name is .npz or .npz, or typo in the extension).

import os
output_dir = '/Users/hosseinsamaei/phd/gemf_llm/output'
files = os.listdir(output_dir)
files
# Previous code assumed the file is named 'agg-static-network-npz'
# The actual file is named 'agg-static-network-npz.npz'
# Fix this in the FastGEMF simulation for scenario 2 and re-run.

import fastgemf as fg
import scipy.sparse as sparse
import os
import numpy as np
import pandas as pd

N = 1000
beta = 0.0047548856450002385  # as from analytic mapping
gamma = 1.0
nsim = 100
agg_network_path = '/Users/hosseinsamaei/phd/gemf_llm/output/agg-static-network-npz.npz'
G_csr = sparse.load_npz(agg_network_path)

SIR_model = (
    fg.ModelSchema('SIR')
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(
        name='recovery1', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network_layer', rate='beta')
)

SIR_instance = (
    fg.ModelConfiguration(SIR_model)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact_network_layer=G_csr)
)

np.random.seed(2)
first_infected = np.random.choice(N)
X0 = np.zeros(N, dtype=int)
X0[first_infected] = 1  # state 0=S, 1=I, 2=R
initial_condition = {'exact': X0}

sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 1000}, nsim=nsim)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-12.png'))

time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i in range(state_count.shape[0]):
    simulation_results[f'{SIR_model.compartments[i]}'] = state_count[i, :]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-12.csv'), index=False)

final_size = state_count[2, -1]
pd.DataFrame({'final_size': [final_size]}).to_csv(os.path.join(os.getcwd(), 'output', 'results-12-finalsizes.csv'), index=False)

N_nodes = N
N_edges = G_csr.nnz//2 # undirected
{
    'N_nodes': N_nodes,
    'N_edges': N_edges
}