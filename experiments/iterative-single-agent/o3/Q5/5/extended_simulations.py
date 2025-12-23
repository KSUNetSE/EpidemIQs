
import os, numpy as np, networkx as nx, scipy.sparse as sparse, random, math, pandas as pd, fastgemf as fg
from tqdm import tqdm

out_dir=os.path.join(os.getcwd(),'output')
csr=sparse.load_npz(os.path.join(out_dir,'network.npz'))
N=csr.shape[0]

k_arr=np.array(csr.sum(axis=1)).flatten()
mean_k=k_arr.mean()
second_moment=(k_arr**2).mean()
q=(second_moment-mean_k)/mean_k
print('mean k',mean_k,'q',q)

idx_k10=np.where(k_arr==10)[0]

# SIR model
SIR_schema=(fg.ModelSchema('SIR').define_compartment(['S','I','R']).add_network_layer('layer').add_node_transition('rec','I','R',rate='gamma').add_edge_interaction('inf','S','I',inducer='I',network_layer='layer',rate='beta'))

beta=1.0
gamma=1.0
cfg=(fg.ModelConfiguration(SIR_schema).add_parameter(beta=beta,gamma=gamma).get_networks(layer=csr))

rng=np.random.default_rng(2024)

sim_cases=[
    {'label':'baseline', 'vacc_nodes':np.array([],dtype=int)},
    {'label':'random75', 'vacc_nodes':rng.choice(N,int(0.75*N),replace=False)},
    {'label':'degree10', 'vacc_nodes':idx_k10}
]

results_summary=[]

for j,case in enumerate(sim_cases,1):
    nsim=50
    metrics={'final_size':[], 'peak_prev':[], 'peak_time':[], 'duration':[]}
    for i in range(nsim):
        X0=np.zeros(N,dtype=int)
        infected=rng.choice(N,10,replace=False)
        X0[infected]=1
        X0[case['vacc_nodes']]=2
        sim=fg.Simulation(cfg,initial_condition={'exact':X0},stop_condition={'time':100},nsim=1)
        sim.run()
        t,sc,*_=sim.get_results()
        I=sc[SIR_schema.compartments.index('I'),:]
        R=sc[SIR_schema.compartments.index('R'),:]
        final_size=R[-1]/N
        peak_prev=I.max()/N
        peak_time=t[I.argmax()]
        # duration as last time I>=1 then + step
        active=np.where(I>=1)[0]
        duration=t[active[-1]] if len(active)>0 else 0
        metrics['final_size'].append(final_size)
        metrics['peak_prev'].append(peak_prev)
        metrics['peak_time'].append(peak_time)
        metrics['duration'].append(duration)
    df=pd.DataFrame(metrics)
    csv_path=os.path.join(out_dir,f'results-1{j}.csv')
    df.to_csv(csv_path,index=False)
    results_summary.append({'case':case['label'], 'vacc_fraction':len(case['vacc_nodes'])/N, 'mean_final_size':np.mean(metrics['final_size']), 'mean_peak_prev':np.mean(metrics['peak_prev'])})

summary_df=pd.DataFrame(results_summary)
summary_df.to_csv(os.path.join(out_dir,'summary_results.csv'),index=False)
print(summary_df)
