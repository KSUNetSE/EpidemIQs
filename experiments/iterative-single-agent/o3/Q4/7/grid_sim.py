
import fastgemf as fg, scipy.sparse as sparse, os, numpy as np, pandas as pd
output_dir=os.path.join(os.getcwd(),'output')
A=sparse.load_npz(os.path.join(output_dir,'network_layer_A.npz'))
B=sparse.load_npz(os.path.join(output_dir,'network_layer_B.npz'))
# model schema
SIS_schema=(fg.ModelSchema('CompetitiveSIS')
    .define_compartment(['S','I1','I2'])
    .add_network_layer('layerA')
    .add_network_layer('layerB')
    .add_edge_interaction('inf1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
    .add_edge_interaction('inf2',from_state='S',to_state='I2',inducer='I2',network_layer='layerB',rate='beta2')
    .add_node_transition('rec1',from_state='I1',to_state='S',rate='delta1')
    .add_node_transition('rec2',from_state='I2',to_state='S',rate='delta2'))

betas=[0.04,0.06,0.08,0.1,0.12]
betas2=[0.06,0.08,0.1,0.12,0.14]
res_list=[]
for b1 in betas:
    for b2 in betas2:
        params={'beta1':b1,'beta2':b2,'delta1':1.0,'delta2':1.0}
        config=(fg.ModelConfiguration(SIS_schema)
            .add_parameter(**params)
            .get_networks(layerA=A,layerB=B))
        init={'percentage':{'I1':1,'I2':1,'S':98}}
        sim=fg.Simulation(config,initial_condition=init,stop_condition={'time':600},nsim=1)
        sim.run()
        time,state_count,*_=sim.get_results()
        final_I1=state_count[1,-1]
        final_I2=state_count[2,-1]
        res_list.append({'beta1':b1,'beta2':b2,'I1_end':final_I1,'I2_end':final_I2})

res_df=pd.DataFrame(res_list)
print(res_df)
