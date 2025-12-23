
# Reasoning for Step 2 (TA-from-ADT, Model #2):
# - Load adjacency from ta-from-adt-aggr-adjacency.npz (aggregated 5 days of ADT)
# - SIR params: beta=0.3, gamma=0.2, N=1000
# - 100 stochastic realizations; 5 random infected
# - Use FastGEMF as for Model #1 for direct comparability
import os
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd

nsim = 100
max_time = 50  # days

# Load TA-from-ADT adjacency
ta_from_adt_path = os.path.join(os.getcwd(), 'output', 'ta-from-adt-aggr-adjacency.npz')
G_taADT = sparse.load_npz(ta_from_adt_path)
N = G_taADT.shape[0]

# Define SIR model
SIR_schema = (
    fg.ModelSchema('SIR')
      .define_compartment(['S', 'I', 'R'])
      .add_network_layer('contacts')
      .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
      .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contacts', rate='beta')
)

SIR_instance = (
    fg.ModelConfiguration(SIR_schema)
      .add_parameter(beta=0.3, gamma=0.2)
      .get_networks(contacts=G_taADT)
)

# 5 random initial infected, rest susceptible
ini = np.zeros(N, dtype=int)
infected_idx = np.random.choice(N, 5, replace=False)
ini[infected_idx] = 1
init_condition = {'exact': ini}

sim = fg.Simulation(SIR_instance, initial_condition=init_condition, stop_condition={'time': max_time}, nsim=nsim)
sim.run()
variation_type = '90ci'
time, statecount, statecount_lbub = sim.get_results(variation_type=variation_type)

# Store CSV
result = {'time': time}
cmps = SIR_schema.compartments
for i in range(statecount.shape[0]):
    result[f"{cmps[i]}"] = statecount[i]
    result[f"{cmps[i]}_{variation_type}_lower"] = statecount_lbub[0, i]
    result[f"{cmps[i]}_{variation_type}_upper"] = statecount_lbub[1, i]
res_df = pd.DataFrame(result)
res_df.to_csv(os.path.join(os.getcwd(), 'output', 'results-12.csv'), index=False)

# Save plot
sim.plot_results(time, statecount, variation_type=variation_type, show_figure=False, save_figure=True, title="SIR on TA-from-ADT", save_path=os.path.join(os.getcwd(), "output", "results-12.png"))
