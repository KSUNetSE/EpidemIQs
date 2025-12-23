
import os, numpy as np, scipy.sparse as sparse, fastgemf as fg, pandas as pd
output_dir = os.path.join(os.getcwd(), 'output')
A_csr = sparse.load_npz(os.path.join(output_dir, 'network_A.npz'))
B_low = sparse.load_npz(os.path.join(output_dir, 'network_B_lowcorr.npz'))

beta1=0.6
beta2=0.6
delta1=delta2=1.0
schema=(fg.ModelSchema('CompetitiveSIS')
        .define_compartment(['S','I1','I2'])
        .add_network_layer('A')
        .add_network_layer('B')
        .add_edge_interaction('inf1', from_state='S', to_state='I1', inducer='I1', network_layer='A', rate='beta1')
        .add_edge_interaction('inf2', from_state='S', to_state='I2', inducer='I2', network_layer='B', rate='beta2')
        .add_node_transition('rec1', from_state='I1', to_state='S', rate='delta1')
        .add_node_transition('rec2', from_state='I2', to_state='S', rate='delta2'))
config=(fg.ModelConfiguration(schema)
        .add_parameter(beta1=beta1, beta2=beta2, delta1=delta1, delta2=delta2)
        .get_networks(A=A_csr, B=B_low))
N=A_csr.shape[0]
rand=np.random.RandomState(777)
perm=rand.permutation(N)
state=np.zeros(N,dtype=int)
state[perm[:int(0.05*N)]]=1
state[perm[int(0.05*N):int(0.1*N)]]=2
init={'exact':state}

sim=fg.Simulation(config, initial_condition=init, stop_condition={'time':600}, nsim=5)
sim.run()
fig_path=os.path.join(output_dir,'results-24.png')
sim.plot_results(show_figure=False, save_figure=True, save_path=fig_path)
time, state_count,*_=sim.get_results()
df=pd.DataFrame({'time':time,'S':state_count[0],'I1':state_count[1],'I2':state_count[2]})
csv_path=os.path.join(output_dir,'results-24.csv')
df.to_csv(csv_path,index=False)
final={'I1':int(state_count[1,-1]),'I2':int(state_count[2,-1])}
