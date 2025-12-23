
import os, scipy.sparse as sparse, fastgemf as fg, numpy as np, pandas as pd
A=sparse.load_npz(os.path.join(os.getcwd(),'output','layer_A.npz'))
B4=sparse.load_npz(os.path.join(os.getcwd(),'output','layer_B4.npz'))

beta1=0.10401059578111396
beta2=0.10401059578111396  # equal rates for test

delta1=delta2=1.0

model_schema=(fg.ModelSchema('SI1I2S')
    .define_compartment(['S','I1','I2'])
    .add_network_layer('layerA')
    .add_network_layer('layerB4')
    .add_edge_interaction('inf1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
    .add_edge_interaction('inf2',from_state='S',to_state='I2',inducer='I2',network_layer='layerB4',rate='beta2')
    .add_node_transition('rec1',from_state='I1',to_state='S',rate='delta1')
    .add_node_transition('rec2',from_state='I2',to_state='S',rate='delta2') )

model_cfg=(fg.ModelConfiguration(model_schema)
    .add_parameter(beta1=beta1,beta2=beta2,delta1=1.0,delta2=1.0)
    .get_networks(layerA=A,layerB4=B4))

n=A.shape[0]
I1_nodes=np.random.choice(n,int(0.05*n),False)
remaining=list(set(range(n))-set(I1_nodes))
I2_nodes=np.random.choice(remaining,int(0.05*n),False)
X0=np.zeros(n,int)
X0[I1_nodes]=1
X0[I2_nodes]=2
initial={'exact':X0}

sim=fg.Simulation(model_cfg,initial_condition=initial,stop_condition={'time':400},nsim=1)

sim.run()
import matplotlib
sim.plot_results(show_figure=False,save_figure=True,save_path=os.path.join(os.getcwd(),'output','results-13.png'))

time,counts,*_=sim.get_results()
final_I1=counts[1,-1]
final_I2=counts[2,-1]
final_I1,final_I2