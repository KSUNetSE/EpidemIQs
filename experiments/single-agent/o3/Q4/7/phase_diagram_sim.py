
import fastgemf as fg, scipy.sparse as sparse, os, numpy as np, pandas as pd, itertools, random
output_dir=os.path.join(os.getcwd(),'output')
A=sparse.load_npz(os.path.join(output_dir,'network_layer_A.npz'))
B=sparse.load_npz(os.path.join(output_dir,'network_layer_B.npz'))

schema=(fg.ModelSchema('CompetitiveSIS').define_compartment(['S','I1','I2'])
         .add_network_layer('layerA')
         .add_network_layer('layerB')
         .add_edge_interaction('inf1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
         .add_edge_interaction('inf2',from_state='S',to_state='I2',inducer='I2',network_layer='layerB',rate='beta2')
         .add_node_transition('rec1',from_state='I1',to_state='S',rate='delta1')
         .add_node_transition('rec2',from_state='I2',to_state='S',rate='delta2'))

beta1_vals=[0.06,0.08,0.1,0.12,0.15,0.18]
beta2_vals=[0.08,0.1,0.12,0.14,0.17,0.2]
result_rows=[]
for b1,b2 in itertools.product(beta1_vals,beta2_vals):
    survive_counts={'I1':0,'I2':0,'both':0}
    reps=5
    final_I1s=[];final_I2s=[]
    for r in range(reps):
        config=(fg.ModelConfiguration(schema)
                 .add_parameter(beta1=b1,beta2=b2,delta1=1.0,delta2=1.0)
                 .get_networks(layerA=A,layerB=B))
        init={'percentage':{'I1':1,'I2':1,'S':98}}
        sim=fg.Simulation(config,initial_condition=init,stop_condition={'time':500},nsim=1)
        sim.run()
        _,state_count,*_=sim.get_results()
        I1_end=state_count[1,-1]
        I2_end=state_count[2,-1]
        final_I1s.append(I1_end)
        final_I2s.append(I2_end)
    avg_I1=np.mean(final_I1s)
    avg_I2=np.mean(final_I2s)
    result_rows.append({'beta1':b1,'beta2':b2,'avg_I1':avg_I1,'avg_I2':avg_I2})

phase_df=pd.DataFrame(result_rows)
print(phase_df.head())
