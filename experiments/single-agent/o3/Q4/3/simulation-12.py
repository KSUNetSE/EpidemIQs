
import os, numpy as np, fastgemf as fg, scipy.sparse as sparse, pandas as pd
from scipy.sparse import load_npz

A_csr = load_npz(os.path.join(os.getcwd(),'output','layerA.npz'))
B_csr = load_npz(os.path.join(os.getcwd(),'output','layerB.npz'))

comp_model = (
    fg.ModelSchema("CompSIS")
    .define_compartment(["S","I1","I2"])
    .add_network_layer('layerA')
    .add_network_layer('layerB')
    .add_edge_interaction(name='adopt1', from_state='S', to_state='I1', inducer='I1', network_layer='layerA', rate='beta1')
    .add_edge_interaction(name='adopt2', from_state='S', to_state='I2', inducer='I2', network_layer='layerB', rate='beta2')
    .add_node_transition(name='rec1', from_state='I1', to_state='S', rate='delta1')
    .add_node_transition(name='rec2', from_state='I2', to_state='S', rate='delta2')
)

params={'beta1':0.11,'beta2':0.11,'delta1':1.0,'delta2':1.0}
model_cfg=(fg.ModelConfiguration(comp_model)
            .add_parameter(**params)
            .get_networks(layerA=A_csr, layerB=B_csr))

N=A_csr.shape[0]
X0=np.zeros(N,dtype=int)
# Random 5% for each infection
rand_idx=np.random.permutation(N)
num= int(0.05*N)
X0[rand_idx[:num]]=1
X0[rand_idx[num:2*num]]=2

initial_condition={'exact':X0}

sim=fg.Simulation(model_cfg, initial_condition=initial_condition, stop_condition={'time':500}, nsim=3)
sim.run()

sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-12.png'))

time, state_count, *_=sim.get_results()
results={'time':time};
for i,comp in enumerate(comp_model.compartments):
    results[comp]=state_count[i,:]
import pandas as pd
pd.DataFrame(results).to_csv(os.path.join(os.getcwd(),'output','results-12.csv'), index=False)

import fastgemf as fg, os, scipy.sparse as sparse, pandas as pd, numpy as np
output_dir=os.path.join(os.getcwd(),'output')
A=sparse.load_npz(os.path.join(output_dir,'network_A.npz'))
B=sparse.load_npz(os.path.join(output_dir,'network_B2.npz'))

schema=(fg.ModelSchema('SI1I2S')
        .define_compartment(['S','I1','I2'])
        .add_network_layer('layerA')
        .add_network_layer('layerB')
        .add_edge_interaction(name='inf1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
        .add_edge_interaction(name='inf2',from_state='S',to_state='I2',inducer='I2',network_layer='layerB',rate='beta2')
        .add_node_transition(name='rec1',from_state='I1',to_state='S',rate='delta1')
        .add_node_transition(name='rec2',from_state='I2',to_state='S',rate='delta2'))
params={'beta1':0.15,'beta2':0.12,'delta1':1.0,'delta2':1.0}
config=(fg.ModelConfiguration(schema).add_parameter(**params).get_networks(layerA=A,layerB=B))
initial={'percentage':{'I1':5,'I2':5,'S':90}}

sim=fg.Simulation(config,initial_condition=initial,stop_condition={'time':300},nsim=5)
sim.run()

sim.plot_results(show_figure=False,save_figure=True,save_path=os.path.join(output_dir,'results-12.png'))

results=sim.get_results()
time=results[0]; counts=results[1]
res={'time':time}
for idx,comp in enumerate(schema.compartments):
    res[comp]=counts[idx,:]

df=pd.DataFrame(res)
df.to_csv(os.path.join(output_dir,'results-12.csv'),index=False)
