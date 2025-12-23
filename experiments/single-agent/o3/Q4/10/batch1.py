
import os, scipy.sparse as sparse, numpy as np, pandas as pd, fastgemf as fg, random
from tqdm import trange

A=sparse.load_npz(os.path.join(os.getcwd(),'output','layer_A.npz'))
B=sparse.load_npz(os.path.join(os.getcwd(),'output','layer_B.npz'))

beta1=0.10401059578111396
beta2=0.13639308317445792
delta1=1.0
delta2=1.0
model_schema=(fg.ModelSchema('SI1I2S')
    .define_compartment(['S','I1','I2'])
    .add_network_layer('layerA')
    .add_network_layer('layerB')
    .add_edge_interaction('infection1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
    .add_edge_interaction('infection2',from_state='S',to_state='I2',inducer='I2',network_layer='layerB',rate='beta2')
    .add_node_transition('rec1',from_state='I1',to_state='S',rate='delta1')
    .add_node_transition('rec2',from_state='I2',to_state='S',rate='delta2')
)

model_config=(fg.ModelConfiguration(model_schema)
    .add_parameter(beta1=beta1,beta2=beta2,delta1=delta1,delta2=delta2)
    .get_networks(layerA=A,layerB=B)
)

n=A.shape[0]
results=[]
for sim_id in trange(10):
    I1_nodes=np.random.choice(n,int(0.05*n),False)
    remaining=list(set(range(n))-set(I1_nodes))
    I2_nodes=np.random.choice(remaining,int(0.05*n),False)
    X0=np.zeros(n,int)
    X0[I1_nodes]=1
    X0[I2_nodes]=2
    initial={'exact':X0}
    sim=fg.Simulation(model_config,initial_condition=initial,stop_condition={'time':300},nsim=1)
    sim.run()
    time,state_count,*_=sim.get_results()
    results.append(state_count[:,-1])

results=np.array(results)
mean_final=results.mean(axis=0)
mean_final