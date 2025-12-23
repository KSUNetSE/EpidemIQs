
import os, numpy as np, fastgemf as fg, pandas as pd, scipy.sparse as sparse
from scipy.sparse import load_npz
os.makedirs(os.path.join(os.getcwd(),'output'), exist_ok=True)
A_csr=load_npz(os.path.join(os.getcwd(),'output','layerA.npz'))
B_csr=load_npz(os.path.join(os.getcwd(),'output','layerB.npz'))

comp_model=(
    fg.ModelSchema('CompSIS')
    .define_compartment(['S','I1','I2'])
    .add_network_layer('A')
    .add_network_layer('B')
    .add_edge_interaction('adopt1','S','I1','I1','A','beta1')
    .add_edge_interaction('adopt2','S','I2','I2','B','beta2')
    .add_node_transition('rec1','I1','S','delta1')
    .add_node_transition('rec2','I2','S','delta2')
)

N=A_csr.shape[0]
beta_vals=[0.08,0.1,0.12,0.14]
results=[]
for b1 in beta_vals:
    for b2 in beta_vals:
        params={'beta1':b1,'beta2':b2,'delta1':1.0,'delta2':1.0}
        cfg=(fg.ModelConfiguration(comp_model)
                .add_parameter(**params)
                .get_networks(A=A_csr,B=B_csr))
        # initial 5% each random
        X0=np.zeros(N,dtype=int)
        idx=np.random.permutation(N)
        num=int(0.05*N)
        X0[idx[:num]]=1
        X0[idx[num:2*num]]=2
        initial={'exact':X0}
        sim=fg.Simulation(cfg, initial_condition=initial, stop_condition={'time':500}, nsim=1)
        sim.run()
        time, state_count,*_=sim.get_results()
        final_I1=state_count[1,-1]
        final_I2=state_count[2,-1]
        coexist=(final_I1>0 and final_I2>0)
        results.append({'beta1':b1,'beta2':b2,'I1_final':final_I1,'I2_final':final_I2,'coexist':coexist})

pd.DataFrame(results).to_csv(os.path.join(os.getcwd(),'output','sweep_results.csv'), index=False)
