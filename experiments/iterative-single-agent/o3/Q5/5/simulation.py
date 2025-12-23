
import os, numpy as np, networkx as nx, scipy.sparse as sparse, fastgemf as fg, random, math, pandas as pd
N=10000
z=3
# Generate network with Poisson degree (ER variant) by configuration model:
# sample degrees from Poisson(3)
rng=np.random.default_rng(0)
ks=rng.poisson(z,size=N)
ks[ks==0]=1 # ensure connectedness maybe
while ks.sum()%2==1:
    ks[0]+=1
G=nx.configuration_model(ks,seed=1)
G=nx.Graph(G)
G.remove_edges_from(nx.selfloop_edges(G))
print('Network created: nodes',G.number_of_nodes(),'edges',G.number_of_edges())

k_arr=np.array([d for _,d in G.degree()])
mean_k=k_arr.mean()
q=(k_arr**2).mean()/mean_k -1
print('mean k',mean_k,'q',q)
# Save
out_dir=os.path.join(os.getcwd(),'output'); os.makedirs(out_dir,exist_ok=True)
csr=nx.to_scipy_sparse_array(G,format='csr')
sparse.save_npz(os.path.join(out_dir,'network_sim.npz'),csr)

# Define SIR with beta=1, gamma=1 so R0=q ~4
SIR_schema=(fg.ModelSchema('SIR').define_compartment(['S','I','R']).add_network_layer('layer').add_node_transition(name='rec',from_state='I',to_state='R',rate='gamma').add_edge_interaction(name='inf',from_state='S',to_state='I',inducer='I',network_layer='layer',rate='beta'))
params={'beta':1.0,'gamma':1.0}

cfg=(fg.ModelConfiguration(SIR_schema).add_parameter(**params).get_networks(layer=csr))

# initial infection: 10 random nodes
init_rand=rng.choice(N,10,replace=False)
X0=np.zeros(N,dtype=int)
X0[init_rand]=1
initial={'exact':X0}

# baseline no vaccination
sim0=fg.Simulation(cfg,initial_condition=initial,stop_condition={'time':50},nsim=1)
sim0.run()

# random vaccination 75%
X1=X0.copy()
rand_nodes=rng.choice(np.arange(N),int(0.75*N),replace=False)
X1[rand_nodes]=2 # mark R state as vaccinated(immune) treat as Recovered
sim1=fg.Simulation(cfg,initial_condition={'exact':X1},stop_condition={'time':50},nsim=1)
sim1.run()

# degree 10 targeted vaccinate all such nodes
X2=X0.copy()
idx_k10=np.where(k_arr==10)[0]
X2[idx_k10]=2
sim2=fg.Simulation(cfg,initial_condition={'exact':X2},stop_condition={'time':50},nsim=1)
sim2.run()

# Extract results

def get_final_size(sim):
    t,sc,*_=sim.get_results()
    return sc[SIR_schema.compartments.index('R'),-1]/N

sizes={'Baseline':get_final_size(sim0), 'Random75%':get_final_size(sim1), 'Target_k10':get_final_size(sim2), 'k10_fraction': len(idx_k10)/N}
