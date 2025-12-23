
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os, pandas as pd
output_dir = os.path.join(os.getcwd(), 'output')
G_er = sparse.load_npz(os.path.join(output_dir, 'network_ER.npz'))
G_ba = sparse.load_npz(os.path.join(output_dir, 'network_BA.npz'))
SEIR_schema = (
    fg.ModelSchema('SEIR')
    .define_compartment(['S','E','I','R'])
    .add_network_layer('contact')
    .add_edge_interaction(name='infection', from_state='S', to_state='E', inducer='I', network_layer='contact', rate='beta')
    .add_node_transition(name='progress', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(name='recover', from_state='I', to_state='R', rate='gamma')
)
params = {'beta':0.05,'sigma':1/4,'gamma':1/7}
config_er = (fg.ModelConfiguration(SEIR_schema).add_parameter(**params).get_networks(contact=G_er))
config_ba = (fg.ModelConfiguration(SEIR_schema).add_parameter(**params).get_networks(contact=G_ba))
N=G_er.shape[0]
initial_condition={'percentage':{'I':(10*100/N),'S':100-(10*100/N),'E':0,'R':0}}
stop={'time':200}
# ER
sim_er = fg.Simulation(config_er, initial_condition=initial_condition, stop_condition=stop, nsim=1)
sim_er.run()
# extract
t, counts, *_ = sim_er.get_results()
res={'time':t}
for i,c in enumerate(SEIR_schema.compartments):
    res[c]=counts[i,:]
pd.DataFrame(res).to_csv(os.path.join(output_dir,'results-11.csv'), index=False)
# BA
sim_ba = fg.Simulation(config_ba, initial_condition=initial_condition, stop_condition=stop, nsim=1)
sim_ba.run()

t2, counts2, *_ = sim_ba.get_results()
res2={'time':t2}
for i,c in enumerate(SEIR_schema.compartments):
    res2[c]=counts2[i,:]
pd.DataFrame(res2).to_csv(os.path.join(output_dir,'results-12.csv'), index=False)
