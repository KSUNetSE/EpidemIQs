
import os, fastgemf as fg, scipy.sparse as sparse, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs(os.path.join(os.getcwd(),'output'),exist_ok=True)
G=sparse.load_npz(os.path.join(os.getcwd(),'output','network.npz'))
SIR=fg.ModelSchema('SIR').define_compartment(['S','I','R']).add_network_layer('c')\
    .add_edge_interaction('inf',from_state='S', to_state='I', inducer='I', network_layer='c', rate='beta')\
    .add_node_transition('rec', from_state='I', to_state='R', rate='gamma')
# network moments
k_mean=9.976; k2=108.922
q=(k2-k_mean)/k_mean
param_sets=[{'R0':0.8,'gamma':0.2,'index':1},{'R0':3.0,'gamma':0.2,'index':2}]
outputs=[]
for p in param_sets:
    beta=p['R0']*p['gamma']/q
    mc=fg.ModelConfiguration(SIR).add_parameter(beta=beta, gamma=p['gamma']).get_networks(c=G)
    initial={'percentage':{'I':1,'S':99}}
    sim=fg.Simulation(mc, initial_condition=initial, stop_condition={'time':160}, nsim=1)
    sim.run()
    time, state_count, *_ = sim.get_results()
    data={'time':time}
    for idx,comp in enumerate(['S','I','R']):
        data[comp]=state_count[idx,:]
    df=pd.DataFrame(data)
    csv_path=os.path.join(os.getcwd(),'output',f'results-1{p["index"]}.csv')
    df.to_csv(csv_path,index=False)
    plt.figure()
    for comp in ['S','I','R']:
        plt.plot(df['time'], df[comp], label=comp)
    plt.legend(); plt.xlabel('Time'); plt.ylabel('Count'); plt.title(f'SIR simulation R0={p["R0"]}')
    png_path=os.path.join(os.getcwd(),'output',f'results-1{p["index"]}.png')
    plt.savefig(png_path); plt.close()
    outputs.append({'csv':csv_path,'png':png_path,'beta':beta,'R0':p['R0']})
print(outputs)
