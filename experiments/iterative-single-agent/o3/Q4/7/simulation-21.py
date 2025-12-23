
import os, itertools, numpy as np, pandas as pd, fastgemf as fg, scipy.sparse as sparse, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
results=[]
Nsim=3
threshold=5 # nodes
for b1,b2 in itertools.product(beta1_vals,beta2_vals):
    I1_list=[];I2_list=[]
    for r in range(Nsim):
        config=(fg.ModelConfiguration(schema)
                 .add_parameter(beta1=b1,beta2=b2,delta1=1.0,delta2=1.0)
                 .get_networks(layerA=A,layerB=B))
        init={'percentage':{'I1':1,'I2':1,'S':98}}
        sim=fg.Simulation(config,initial_condition=init,stop_condition={'time':500},nsim=1)
        sim.run()
        _,sc,*_=sim.get_results()
        I1_list.append(sc[1,-1])
        I2_list.append(sc[2,-1])
    avg_I1=np.mean(I1_list)
    avg_I2=np.mean(I2_list)
    # categorize
    if avg_I1>threshold and avg_I2>threshold:
        outcome='coexist'
    elif avg_I1>threshold and avg_I2<=threshold:
        outcome='I1_dom'
    elif avg_I2>threshold and avg_I1<=threshold:
        outcome='I2_dom'
    else:
        outcome='extinct'
    results.append({'beta1':b1,'beta2':b2,'avg_I1':avg_I1,'avg_I2':avg_I2,'outcome':outcome})

phase_df=pd.DataFrame(results)
phase_df.to_csv(os.path.join(output_dir,'results-21.csv'),index=False)
# Plot phase diagram
fig,ax=plt.subplots(figsize=(6,5))
outcome_colors={'coexist':'purple','I1_dom':'blue','I2_dom':'red','extinct':'grey'}
for outcome,grp in phase_df.groupby('outcome'):
    ax.scatter(grp['beta1'],grp['beta2'],c=outcome_colors[outcome],label=outcome,s=80,edgecolors='k')
ax.set_xlabel('beta1')
ax.set_ylabel('beta2')
ax.set_title('Phase diagram of competition outcomes')
ax.legend()
fig.savefig(os.path.join(output_dir,'results-21.png'))
plt.close(fig)
