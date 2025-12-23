
# Reasoning for Step 1 (TA-ER, Model #1):
# - Load ER-like static network (adjacency from ta-er-statich-graph-adjacency.npz)
# - SIR params: beta=0.3, gamma=0.2, N=1000
# - Initial: 5 randomly-chosen infected (1%), rest susceptible
# - 100 stochastic realizations, 50+ days till fade-out
# - Use FastGEMF (static network, matches requirements)
import os
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd

# Set realization count and simulation stop time
nsim = 100
max_time = 50  # days; more than 10x infectious period

# 1. Load static ER network adjacency matrix (check and load)
er_npz = os.path.join(os.getcwd(), 'output', 'ta-er-statich-graph-adjacency.npz')
G_csr = sparse.load_npz(er_npz)
N = G_csr.shape[0]

# 2. Model: SIR with correct transitions
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contacts')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contacts', rate='beta')
)

# 3. Model Configuration
SIR_instance = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=0.3, gamma=0.2)
    .get_networks(contacts=G_csr)
)

# 4. Initial condition: 5 infected (randomly-chosen); rest susceptible for N=1000
ini = np.zeros(N, dtype=int)  # 0: S, 1: I, 2: R
infected_idx = np.random.choice(N, 5, replace=False)
ini[infected_idx] = 1

init_condition = {'exact': ini}

# 5. Simulate SIR using FastGEMF
sim = fg.Simulation(SIR_instance, initial_condition=init_condition, stop_condition={'time': max_time}, nsim=nsim)
sim.run()
variation_type = "90ci"
time, statecount, statecount_lbub = sim.get_results(variation_type=variation_type)

# 6. Store results (CSV)
result = {'time': time}
cmps = SIR_schema.compartments
for i in range(statecount.shape[0]):
    result[f"{cmps[i]}"] = statecount[i]
    result[f"{cmps[i]}_{variation_type}_lower"] = statecount_lbub[0, i]
    result[f"{cmps[i]}_{variation_type}_upper"] = statecount_lbub[1, i]
res_df = pd.DataFrame(result)
res_df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

# 7. Save plot
sim.plot_results(time, statecount, variation_type=variation_type, show_figure=False, save_figure=True, title="SIR on time-aggregated ER", save_path=os.path.join(os.getcwd(), "output", "results-11.png"))
