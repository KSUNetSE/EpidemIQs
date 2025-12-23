
import os, numpy as np, scipy.sparse as sparse
import fastgemf as fg
import pandas as pd, matplotlib

# Load network
G_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

# Define SIR schema
SIR_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S','I','R'])
    .add_network_layer('contact')
    .add_edge_interaction('infection', from_state='S', to_state='I', inducer='I', network_layer='contact', rate='beta')
    .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
)

# Scenario 1: beta_high
params1 = {'beta':0.03,'gamma':0.1}
config1 = (fg.ModelConfiguration(SIR_schema).add_parameter(**params1).get_networks(contact=G_csr))

IC = {'hubs_number': {'I':10, 'S':990}}  # 10 hubs as infected maybe default picks high-degree nodes

sim1 = fg.Simulation(config1, initial_condition=IC, stop_condition={'time':160}, nsim=1)
sim1.run()
# Save fig
sim1.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-11.png'))
# Save data
import matplotlib.pyplot as plt

time, counts, *_ = sim1.get_results()
res_df = pd.DataFrame({'time':time, 'S':counts[0], 'I':counts[1], 'R':counts[2]})
res_df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

# Scenario 2: beta_low
params2 = {'beta':0.005, 'gamma':0.1}
config2 = (fg.ModelConfiguration(SIR_schema).add_parameter(**params2).get_networks(contact=G_csr))
sim2 = fg.Simulation(config2, initial_condition=IC, stop_condition={'time':160}, nsim=1)
sim2.run()
sim2.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-12.png'))

time2, counts2, *_ = sim2.get_results()
res_df2 = pd.DataFrame({'time':time2, 'S':counts2[0], 'I':counts2[1], 'R':counts2[2]})
res_df2.to_csv(os.path.join(os.getcwd(), 'output', 'results-12.csv'), index=False)
