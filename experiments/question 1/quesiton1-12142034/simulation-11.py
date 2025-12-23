
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

# ---- SCENARIO 1: ER NETWORK (i=1, j=1) ----
# 1. Load ER network
er_net_path = '/Users/hosseinsamaei/phd/epidemiqs/output/er-network.npz'
G_er_csr = sparse.load_npz(er_net_path)

# 2. Define classic SEIR schema
seir_schema_er = (
    fg.ModelSchema('SEIR')
    .define_compartment(['S','E','I','R'])
    .add_network_layer('contact')
    .add_node_transition(name='progress', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(name='recover', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infect', from_state='S', to_state='E', inducer='I', network_layer='contact', rate='beta')
)

# 3. Model configuration (rates per scenario description)
seir_config_er = (
    fg.ModelConfiguration(seir_schema_er)
    .add_parameter(beta=0.0383, sigma=0.333, gamma=0.25)
    .get_networks(contact=G_er_csr)
)

# 4. Initial condition: 1% E, 99% S
ic_er = {'percentage': {'E': 1, 'S': 99}}

# 5. Run simulation
sim_er = fg.Simulation(seir_config_er, initial_condition=ic_er, stop_condition={'time': end_time}, nsim=sr)
sim_er.run()
time_er, statecount_er, (lb_er, ub_er) = sim_er.get_results(variation_type=variation_type)

# 6. Collect & save results
resdict_er = {'time': time_er}
comps = seir_schema_er.compartments
for idx, c in enumerate(comps):
    resdict_er[f'{c}'] = statecount_er[idx]
    resdict_er[f'{c}_{variation_type}_lower'] = lb_er[idx]
    resdict_er[f'{c}_{variation_type}_upper'] = ub_er[idx]
data_er = pd.DataFrame(resdict_er)
out_csv_er = os.path.join(os.getcwd(), 'output','results-11.csv')
data_er.to_csv(out_csv_er, index=False)

# 7. Plot
out_png_er = os.path.join(os.getcwd(), 'output', 'results-11.png')
sim_er.plot_results(time_er, statecount_er, variation_type=variation_type, show_figure=False, save_figure=True, title='SEIR on ER network', save_path=out_png_er)

outputs_er = {
    'csv': out_csv_er,
    'png': out_png_er,
    'stats': {
        'peak_I': int(statecount_er[comps.index('I')].max()),
        'final_R': int(statecount_er[comps.index('R')][-1])
    }
}
