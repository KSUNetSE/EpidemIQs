
import fastgemf as fg, scipy.sparse as sparse, os, numpy as np, pandas as pd
out=os.path.join(os.getcwd(),'output')
G_er=sparse.load_npz(os.path.join(out,'network_ER.npz'))
G_ba=sparse.load_npz(os.path.join(out,'network_BA.npz'))
SEIR_schema=(fg.ModelSchema('SEIR').define_compartment(['S','E','I','R']).add_network_layer('contact').add_edge_interaction(name='inf',from_state='S',to_state='E',inducer='I',network_layer='contact',rate='beta').add_node_transition(name='prog',from_state='E',to_state='I',rate='sigma').add_node_transition(name='rec',from_state='I',to_state='R',rate='gamma'))

sigma=1/4
gamma=1/7
N=G_er.shape[0]
perc_I=10*100/N
init={'percentage':{'I':perc_I,'S':100-perc_I,'E':0,'R':0}}
stop={'time':160}
nsim=20
beta=0.05
metrics=[]
for net_name,G in [('ER',G_er),('BA',G_ba)]:
    for run in range(nsim):
        config=(fg.ModelConfiguration(SEIR_schema).add_parameter(beta=beta,sigma=sigma,gamma=gamma).get_networks(contact=G))
        sim=fg.Simulation(config,initial_condition=init,stop_condition=stop,nsim=1)
        sim.run()
        t,counts=sim.get_results()
        I=counts[2,:]
        peak_I=float(I.max())
        t_peak=float(t[I.argmax()])
        final_R=float(counts[3,-1])
        duration=int(np.sum(I>1))
        metrics.append({'network':net_name,'peak_I':peak_I,'t_peak':t_peak,'final_R':final_R,'duration':duration})
metrics_df=pd.DataFrame(metrics)
metrics_df.to_csv(os.path.join(out,'metrics_20runs.csv'),index=False)
avg_df=metrics_df.groupby('network').agg({'peak_I':['mean','std'], 't_peak':['mean','std'],'final_R':['mean','std'],'duration':['mean','std']})
avg_df.to_csv(os.path.join(out,'metrics_avg_20runs.csv'))
