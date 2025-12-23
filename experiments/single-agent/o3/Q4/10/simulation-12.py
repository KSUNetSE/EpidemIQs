
import os, numpy as np, fastgemf as fg, scipy.sparse as sparse, pandas as pd
from scipy.sparse import load_npz

A_csr = load_npz(os.path.join(os.getcwd(),'output','layerA.npz'))
B_csr = load_npz(os.path.join(os.getcwd(),'output','layerB.npz'))

comp_model = (
    fg.ModelSchema("CompSIS")
    .define_compartment(["S","I1","I2"])
    .add_network_layer('layerA')
    .add_network_layer('layerB')
    .add_edge_interaction(name='adopt1', from_state='S', to_state='I1', inducer='I1', network_layer='layerA', rate='beta1')
    .add_edge_interaction(name='adopt2', from_state='S', to_state='I2', inducer='I2', network_layer='layerB', rate='beta2')
    .add_node_transition(name='rec1', from_state='I1', to_state='S', rate='delta1')
    .add_node_transition(name='rec2', from_state='I2', to_state='S', rate='delta2')
)

params={'beta1':0.11,'beta2':0.11,'delta1':1.0,'delta2':1.0}
model_cfg=(fg.ModelConfiguration(comp_model)
            .add_parameter(**params)
            .get_networks(layerA=A_csr, layerB=B_csr))

N=A_csr.shape[0]
X0=np.zeros(N,dtype=int)
# Random 5% for each infection
rand_idx=np.random.permutation(N)
num= int(0.05*N)
X0[rand_idx[:num]]=1
X0[rand_idx[num:2*num]]=2

initial_condition={'exact':X0}

sim=fg.Simulation(model_cfg, initial_condition=initial_condition, stop_condition={'time':500}, nsim=3)
sim.run()

sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-12.png'))

time, state_count, *_=sim.get_results()
results={'time':time};
for i,comp in enumerate(comp_model.compartments):
    results[comp]=state_count[i,:]
import pandas as pd
pd.DataFrame(results).to_csv(os.path.join(os.getcwd(),'output','results-12.csv'), index=False)

import fastgemf as fg, os, scipy.sparse as sparse, pandas as pd, numpy as np
output_dir=os.path.join(os.getcwd(),'output')
A=sparse.load_npz(os.path.join(output_dir,'network_A.npz'))
B=sparse.load_npz(os.path.join(output_dir,'network_B2.npz'))

schema=(fg.ModelSchema('SI1I2S')
        .define_compartment(['S','I1','I2'])
        .add_network_layer('layerA')
        .add_network_layer('layerB')
        .add_edge_interaction(name='inf1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
        .add_edge_interaction(name='inf2',from_state='S',to_state='I2',inducer='I2',network_layer='layerB',rate='beta2')
        .add_node_transition(name='rec1',from_state='I1',to_state='S',rate='delta1')
        .add_node_transition(name='rec2',from_state='I2',to_state='S',rate='delta2'))
params={'beta1':0.15,'beta2':0.12,'delta1':1.0,'delta2':1.0}
config=(fg.ModelConfiguration(schema).add_parameter(**params).get_networks(layerA=A,layerB=B))
initial={'percentage':{'I1':5,'I2':5,'S':90}}

sim=fg.Simulation(config,initial_condition=initial,stop_condition={'time':300},nsim=5)
sim.run()

sim.plot_results(show_figure=False,save_figure=True,save_path=os.path.join(output_dir,'results-12.png'))

results=sim.get_results()
time=results[0]; counts=results[1]
res={'time':time}
for idx,comp in enumerate(schema.compartments):
    res[comp]=counts[idx,:]

df=pd.DataFrame(res)
df.to_csv(os.path.join(output_dir,'results-12.csv'),index=False)

import os, numpy as np, scipy.sparse as sparse, pandas as pd, fastgemf as fg
output_dir = os.path.join(os.getcwd(), 'output')
A_csr = sparse.load_npz(os.path.join(output_dir, 'network_A.npz'))
B_csr = sparse.load_npz(os.path.join(output_dir, 'network_B.npz'))

lambda1_A = 14.421607613484786
lambda1_B = 7.362893722018067

delta1 = delta2 = 1.0
beta1 = 3.0 / lambda1_A  # 0.208
beta2 = 3.0 / lambda1_B  # 0.407

SIS2_schema = (
    fg.ModelSchema('CompetitiveSIS')
    .define_compartment(['S', 'I1', 'I2'])
    .add_network_layer('layerA')
    .add_network_layer('layerB')
    .add_edge_interaction('infection1', from_state='S', to_state='I1', inducer='I1', network_layer='layerA', rate='beta1')
    .add_edge_interaction('infection2', from_state='S', to_state='I2', inducer='I2', network_layer='layerB', rate='beta2')
    .add_node_transition('recov1', from_state='I1', to_state='S', rate='delta1')
    .add_node_transition('recov2', from_state='I2', to_state='S', rate='delta2')
)
config = (fg.ModelConfiguration(SIS2_schema)
          .add_parameter(beta1=beta1, beta2=beta2, delta1=delta1, delta2=delta2)
          .get_networks(layerA=A_csr, layerB=B_csr))
# Initial: same as before
N = A_csr.shape[0]
state_array = np.zeros(N, dtype=int)
rand = np.random.RandomState(321)
indices = np.arange(N)
rand.shuffle(indices)
I1_nodes = indices[:int(0.1*N)]  # 10%
I2_nodes = indices[int(0.1*N):int(0.2*N)]
state_array[I1_nodes] = 1
state_array[I2_nodes] = 2
initial_condition = {'exact': state_array}

sim = fg.Simulation(config, initial_condition=initial_condition, stop_condition={'time':500}, nsim=5)
sim.run()
fig_path = os.path.join(output_dir, 'results-12.png')
sim.plot_results(show_figure=False, save_figure=True, save_path=fig_path)
time, state_count, *_ = sim.get_results()
results = {'time': time, 'S': state_count[0], 'I1': state_count[1], 'I2': state_count[2]}
import pandas as pd
pd.DataFrame(results).to_csv(os.path.join(output_dir, 'results-12.csv'), index=False)
final = {'S': state_count[0,-1], 'I1': state_count[1,-1], 'I2': state_count[2,-1]}

import os, scipy.sparse as sparse, numpy as np, networkx as nx, fastgemf as fg, pandas as pd, random

# Load A
A_path=os.path.join(os.getcwd(),'output','layer_A.npz')
A=sparse.load_npz(A_path)
# Create B2 by permuting nodes randomly
n=A.shape[0]
perm=list(range(n))
random.shuffle(perm)
perm_inv=np.argsort(perm)
A_csr=A
# apply permutation to adjacency: B2=PAP^T
from scipy.sparse import csr_matrix
P=csr_matrix((np.ones(n),(perm, range(n))),shape=(n,n))
B2=P@A_csr@P.T
# Save
sparse.save_npz(os.path.join(os.getcwd(),'output','layer_B2.npz'),B2)

# Compute eigenvectors correlation
from scipy.sparse import linalg as spla
va=spla.eigs(A,k=1,which='LR')[1][:,0]
vb=spla.eigs(B2,k=1,which='LR')[1][:,0]
cos=abs((va.conj().T@vb)/(np.linalg.norm(va)*np.linalg.norm(vb)))

beta1=0.10401059578111396
beta2=beta1  # same rate

model_schema=(fg.ModelSchema('SI1I2S')
    .define_compartment(['S','I1','I2'])
    .add_network_layer('layerA')
    .add_network_layer('layerB2')
    .add_edge_interaction('infection1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
    .add_edge_interaction('infection2',from_state='S',to_state='I2',inducer='I2',network_layer='layerB2',rate='beta2')
    .add_node_transition('rec1',from_state='I1',to_state='S',rate='delta1')
    .add_node_transition('rec2',from_state='I2',to_state='S',rate='delta2')
)

model_config=(fg.ModelConfiguration(model_schema)
      .add_parameter(beta1=beta1,beta2=beta2,delta1=1.0,delta2=1.0)
      .get_networks(layerA=A,layerB2=B2)
)

# initial 5% each
n=A.shape[0]
I1_nodes=np.random.choice(n,int(0.05*n),False)
remaining=list(set(range(n))-set(I1_nodes))
I2_nodes=np.random.choice(remaining,int(0.05*n),False)
X0=np.zeros(n,int)
X0[I1_nodes]=1
X0[I2_nodes]=2
initial_condition={'exact':X0}

sim=fg.Simulation(model_config,initial_condition=initial_condition,stop_condition={'time':300},nsim=1)
sim.run()
sim.plot_results(show_figure=False,save_figure=True,save_path=os.path.join(os.getcwd(),'output','results-12.png'))

time,state_count,*_=sim.get_results()
results={'time':time}
for i,comp in enumerate(model_schema.compartments):
    results[comp]=state_count[i,:]

df=pd.DataFrame(results)
path_csv=os.path.join(os.getcwd(),'output','results-12.csv')
df.to_csv(path_csv,index=False)

# metrics
final=df.tail(1)
coexist=(final['I1'].values[0]>0) and (final['I2'].values[0]>0)
metrics2={'evec_cos':cos,'coexist':bool(coexist),'final_I1':int(final['I1']), 'final_I2':int(final['I2'])}
metrics2