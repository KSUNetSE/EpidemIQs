
# Step 5: (Optional) Simulate well-mixed control for qualitative attack rate/peak comparison, using an ER random graph
import networkx as nx
import fastgemf as fg
from scipy import sparse
import numpy as np
import pandas as pd
import os
N = 763
mean_degree = 22.82
p = mean_degree/(N-1)
G_ER = nx.erdos_renyi_graph(N, p, seed=42)
G_csr = nx.to_scipy_sparse_array(G_ER, format='csr')
SEIBR_schema = (
    fg.ModelSchema('SEIBR')
    .define_compartment(['S', 'E', 'I', 'B', 'R'])
    .add_network_layer('contact_network')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='E', inducer='I',
        network_layer='contact_network', rate='beta'
    )
    .add_node_transition(name='latent_exit', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(name='symptom_isolation', from_state='I', to_state='B', rate='gamma1')
    .add_node_transition(name='recover', from_state='B', to_state='R', rate='gamma2')
)
params = {"beta": 0.3789, "sigma": 2.0, "gamma1": 0.7, "gamma2": 0.5}
SEIBR_model = (
    fg.ModelConfiguration(SEIBR_schema)
    .add_parameter(**params)
    .get_networks(contact_network=G_csr)
)
# Initial: one random E
X0 = np.zeros(N, dtype=int)
np.random.seed(42)
seed = np.random.randint(0, N)
X0[seed] = 1
initial_condition = {"exact": X0}
sr = 100
sim = fg.Simulation(SEIBR_model, initial_condition=initial_condition, stop_condition={'time': 50}, nsim=sr)
sim.run()
variation_type = "90ci"
time, state_mean, [state_low, state_high] = sim.get_results(variation_type=variation_type)
out_dict = {'time': time}
comps = ['S', 'E', 'I', 'B', 'R']
for i, c in enumerate(comps):
    out_dict[c] = state_mean[i]
    out_dict[c+'_lower'] = state_low[i]
    out_dict[c+'_upper'] = state_high[i]
C = np.diff(state_mean[4], prepend=0)
out_dict['C'] = C
data = pd.DataFrame(out_dict)
result_path = os.path.join(os.getcwd(), 'output', 'results-16.csv')
data.to_csv(result_path, index=False)
sim.plot_results(time, state_mean, variation_type=variation_type, show_figure=False, save_figure=True,
    title="SEIBR ER simulation: Beta=0.3789 (well-mixed)",
    save_path=os.path.join(os.getcwd(), 'output', 'results-16.png'))
attack_rate = state_mean[4, -1] / N
peak_B_idx = np.argmax(state_mean[3])
peak_B_time = time[peak_B_idx]
peak_B_value = state_mean[3, peak_B_idx]
return_vars = [
    'result_path', 'attack_rate', 'peak_B_time', 'peak_B_value'
]
