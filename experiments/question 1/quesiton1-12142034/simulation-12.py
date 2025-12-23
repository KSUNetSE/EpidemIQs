
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

# SETTINGS
N = 5000
sr = 100  # stochastic realizations
end_time = 365  # days
variation_type = '90ci'  # 90% confidence interval

# ---- SCENARIO 2: BA NETWORK (i=1, j=2) ----
# 1. Load BA network
ba_net_path = '/Users/hosseinsamaei/phd/epidemiqs/output/ba-network.npz'
G_ba_csr = sparse.load_npz(ba_net_path)

# 2. Define classic SEIR schema (as in ER, since degree-stratified model would need extensions)
seir_schema_ba = (
    fg.ModelSchema('SEIR')
    .define_compartment(['S','E','I','R'])
    .add_network_layer('contact')
    .add_node_transition(name='progress', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(name='recover', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infect', from_state='S', to_state='E', inducer='I', network_layer='contact', rate='beta')
)

# 3. Model configuration (rates per scenario description)
seir_config_ba = (
    fg.ModelConfiguration(seir_schema_ba)
    .add_parameter(beta=0.0375, sigma=0.333, gamma=0.25)
    .get_networks(contact=G_ba_csr)
)

# 4. Initial condition: 1% E, 99% S
ic_ba = {'percentage': {'E': 1, 'S': 99}}

# 5. Run simulation
sim_ba = fg.Simulation(seir_config_ba, initial_condition=ic_ba, stop_condition={'time': end_time}, nsim=sr)
sim_ba.run()
time_ba, statecount_ba, (lb_ba, ub_ba) = sim_ba.get_results(variation_type=variation_type)

# 6. Collect & save results
resdict_ba = {'time': time_ba}
comps = seir_schema_ba.compartments
for idx, c in enumerate(comps):
    resdict_ba[f'{c}'] = statecount_ba[idx]
    resdict_ba[f'{c}_{variation_type}_lower'] = lb_ba[idx]
    resdict_ba[f'{c}_{variation_type}_upper'] = ub_ba[idx]
data_ba = pd.DataFrame(resdict_ba)
out_csv_ba = os.path.join(os.getcwd(), 'output','results-12.csv')
data_ba.to_csv(out_csv_ba, index=False)

# 7. Plot
out_png_ba = os.path.join(os.getcwd(), 'output', 'results-12.png')
sim_ba.plot_results(time_ba, statecount_ba, variation_type=variation_type, show_figure=False, save_figure=True, title='SEIR on BA network', save_path=out_png_ba)

outputs_ba = {
    'csv': out_csv_ba,
    'png': out_png_ba,
    'stats': {
        'peak_I': int(statecount_ba[comps.index('I')].max()),
        'final_R': int(statecount_ba[comps.index('R')][-1])
    }
}
