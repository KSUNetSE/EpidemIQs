
import os, networkx as nx, scipy.sparse as sp, random, pandas as pd, numpy as np, matplotlib.pyplot as plt, collections, math, time
random.seed(42)
output_dir=os.path.join(os.getcwd(),'output')
csr=sp.load_npz(os.path.join(output_dir,'network.npz'))
G=nx.from_scipy_sparse_array(csr)
N=G.number_of_nodes()


def simulate_SI(G, max_steps=100):
    # Infect one random node, discrete time steps, p=1 for infection
    infected=set([random.choice(list(G.nodes()))])
    times=[0]
    sizes=[1]
    step=0
    susceptible=set(G.nodes())-infected
    while infected and step<max_steps:
        new_infected=set()
        for node in infected:
            for nbr in G.neighbors(node):
                if nbr in susceptible:
                    new_infected.add(nbr)
        susceptible-=new_infected
        infected=new_infected
        step+=1
        times.append(step)
        sizes.append(len(G.nodes())-len(susceptible))
        if len(sizes[-1:]) and sizes[-1]==N: # all infected
            break
    return times, sizes

# scenario A: random vacc 75%
frac=0.75
rand_nodes=random.sample(list(G.nodes()), int(frac*N))
G_rand=G.copy()
G_rand.remove_nodes_from(rand_nodes)

# run 3 simulations save aggregated mean series length maybe variable; we will run one for demonstration

times, sizes=simulate_SI(G_rand)

# Save to csv
csv_path=os.path.join(output_dir,'results-11.csv')
pd.DataFrame({'time':times,'cum_infected':sizes}).to_csv(csv_path,index=False)

# Plot
plt.figure()
plt.plot(times,sizes)
plt.xlabel('time step')
plt.ylabel('cumulative infected')
plt.title('Random vaccination 75%')
plt.tight_layout()
plt_path=os.path.join(output_dir,'results-11.png')
plt.savefig(plt_path)
plt.close()

# scenario B: vaccinate k=10 nodes
k10_nodes=[n for n,d in G.degree() if d==10]
G_tar=G.copy()
G_tar.remove_nodes_from(k10_nodes)
Btimes,Bsizes=simulate_SI(G_tar)

csv_path2=os.path.join(output_dir,'results-12.csv')
pd.DataFrame({'time':Btimes,'cum_infected':Bsizes}).to_csv(csv_path2,index=False)

plt.figure()
plt.plot(Btimes,Bsizes)
plt.xlabel('time step')
plt.ylabel('cumulative infected')
plt.title('Vaccinate degree 10 nodes')
plt.tight_layout()
plt_path2=os.path.join(output_dir,'results-12.png')
plt.savefig(plt_path2)
plt.close()
return_dict={'rand_len':len(G_rand),'tar_len':len(G_tar)}

import os, sys, numpy as np, scipy.sparse as sparse, networkx as nx, matplotlib.pyplot as plt
from numpy.random import default_rng

def run_simulation(G_A_csr, G_B_csr, beta1, delta1, beta2, delta2, initial_infected1_pct=1, initial_infected2_pct=1, t_max=300, nsim=5, rng_seed=42):
    rng = default_rng(rng_seed)
    N = G_A_csr.shape[0]
    # Precompute neighbor lists for efficiency
    neighbors_A = [G_A_csr.indices[G_A_csr.indptr[i]:G_A_csr.indptr[i+1]] for i in range(N)]
    neighbors_B = [G_B_csr.indices[G_B_csr.indptr[i]:G_B_csr.indptr[i+1]] for i in range(N)]

    results_time = None
    results_S = None
    results_I1 = None
    results_I2 = None

    for sim in range(nsim):
        # 0=Susceptible,1=I1,2=I2
        state = np.zeros(N, dtype=int)
        # infect initial percentages randomly
        idx = rng.permutation(N)
        n1 = int(initial_infected1_pct/100*N)
        n2 = int(initial_infected2_pct/100*N)
        state[idx[:n1]] = 1
        state[idx[n1:n1+n2]] = 2

        t=0.0
        dt=0.1
        time_series=[0]
        S_series=[(state==0).sum()]
        I1_series=[(state==1).sum()]
        I2_series=[(state==2).sum()]
        while t<t_max:
            # For each susceptible node compute infection probability from each meme this step
            # Use mean-field approximation dt small: prob adopt meme1 = 1-exp(-beta1*I_neighbors_A*dt) approx beta1*I_neighbors_A*dt (if small)
            new_state = state.copy()
            for node in range(N):
                if state[node]==0: # susceptible
                    infected1 = False
                    infected2 = False
                    # meme1 exposure
                    for nb in neighbors_A[node]:
                        if state[nb]==1:
                            if rng.random()<beta1*dt:
                                infected1=True
                                break
                    # meme2 exposure only if not infected1 already (exclusive)
                    if not infected1:
                        for nb in neighbors_B[node]:
                            if state[nb]==2:
                                if rng.random()<beta2*dt:
                                    infected2=True
                                    break
                    if infected1:
                        new_state[node]=1
                    elif infected2:
                        new_state[node]=2
                elif state[node]==1:
                    if rng.random()<delta1*dt:
                        new_state[node]=0
                elif state[node]==2:
                    if rng.random()<delta2*dt:
                        new_state[node]=0
            state = new_state
            t+=dt
            if int(t/dt)%10==0:
                time_series.append(t)
                S_series.append((state==0).sum())
                I1_series.append((state==1).sum())
                I2_series.append((state==2).sum())
        # aggregate
        time_arr = np.array(time_series)
        S_arr = np.array(S_series)
        I1_arr = np.array(I1_series)
        I2_arr = np.array(I2_series)
        if results_time is None:
            results_time=time_arr
            results_S=S_arr
            results_I1=I1_arr
            results_I2=I2_arr
        else:
            results_S+=S_arr
            results_I1+=I1_arr
            results_I2+=I2_arr
    # average
    results_S=results_S/nsim
    results_I1=results_I1/nsim
    results_I2=results_I2/nsim
    return results_time, results_S, results_I1, results_I2

output_dir=os.path.join(os.getcwd(),'output')
G_A=sparse.load_npz(os.path.join(output_dir,'layerA.npz'))
G_B1=sparse.load_npz(os.path.join(output_dir,'layerB1.npz'))
G_B2=sparse.load_npz(os.path.join(output_dir,'layerB2.npz'))

lambda1_A=np.linalg.eigvals(G_A.toarray()).real.max()
lambda1_B1=lambda1_A
lambda1_B2=np.linalg.eigvals(G_B2.toarray()).real.max()

# parameters
beta_factor=1.2
beta1=beta_factor/lambda1_A
beta2_identical=beta_factor/lambda1_B1
beta2_uncorr=beta_factor/lambda1_B2

delta1=1.0
delta2=1.0

# run for identical layers case

time1,S1,I1_1,I2_1=run_simulation(G_A,G_B1,beta1,delta1,beta2_identical,delta2)

# run for distinct layers case

time2,S2,I1_2,I2_2=run_simulation(G_A,G_B2,beta1,delta1,beta2_uncorr,delta2)

import pandas as pd

data1=pd.DataFrame({'time':time1,'S':S1,'I1':I1_1,'I2':I2_1})
data2=pd.DataFrame({'time':time2,'S':S2,'I1':I1_2,'I2':I2_2})

# save
results1_path=os.path.join(output_dir,'results-11.csv')
results2_path=os.path.join(output_dir,'results-12.csv')

data1.to_csv(results1_path,index=False)
data2.to_csv(results2_path,index=False)

# plot
plt.figure()
plt.plot(time1,I1_1,label='I1 (identical)')
plt.plot(time1,I2_1,label='I2 (identical)')
plt.xlabel('time')
plt.ylabel('infected')
plt.legend()
plt.savefig(os.path.join(output_dir,'results-11.png'))
plt.close()

plt.figure()
plt.plot(time2,I1_2,label='I1 (uncorr)')
plt.plot(time2,I2_2,label='I2 (uncorr)')
plt.xlabel('time')
plt.ylabel('infected')
plt.legend()
plt.savefig(os.path.join(output_dir,'results-12.png'))
plt.close()


import os, numpy as np, fastgemf as fg, scipy.sparse as sparse, pandas as pd
from scipy.sparse import load_npz

os.makedirs(os.path.join(os.getcwd(),'output'), exist_ok=True)

A_csr = load_npz(os.path.join(os.getcwd(),'output','layerA.npz'))
B_csr = load_npz(os.path.join(os.getcwd(),'output','layerB.npz'))

comp_model = (
    fg.ModelSchema("CompSIS")
    .define_compartment(["S","I1","I2"])
    .add_network_layer('layerA')
    .add_network_layer('layerB')
    .add_edge_interaction(name='adoption1', from_state='S', to_state='I1', inducer='I1', network_layer='layerA', rate='beta1')
    .add_edge_interaction(name='adoption2', from_state='S', to_state='I2', inducer='I2', network_layer='layerB', rate='beta2')
    .add_node_transition(name='recovery1', from_state='I1', to_state='S', rate='delta1')
    .add_node_transition(name='recovery2', from_state='I2', to_state='S', rate='delta2')
)

params={'beta1':0.1,'beta2':0.12,'delta1':1.0,'delta2':1.0}
model_cfg = (
    fg.ModelConfiguration(comp_model)
    .add_parameter(**params)
    .get_networks(layerA=A_csr, layerB=B_csr)
)

N=A_csr.shape[0]
X0=np.zeros(N,dtype=int)

# hubs in layerA
import numpy as np
degrees = np.array(A_csr.sum(axis=1)).flatten()
num_hubs=int(0.05*N)
idx_hubs=np.argsort(-degrees)[:num_hubs]
X0[idx_hubs]=1

remaining=np.setdiff1d(np.arange(N), idx_hubs)
num_i2=int(0.05*N)
idx_i2=np.random.choice(remaining, num_i2, replace=False)
X0[idx_i2]=2

initial_condition={'exact':X0}

sim=fg.Simulation(model_cfg, initial_condition=initial_condition, stop_condition={'time':500}, nsim=3)

sim.run()

sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-11.png'))

time,state_count,*_=sim.get_results()
results={'time':time}
for i,comp in enumerate(comp_model.compartments):
    results[comp]=state_count[i,:]

pd.DataFrame(results).to_csv(os.path.join(os.getcwd(),'output','results-11.csv'), index=False)

import fastgemf as fg, os, scipy.sparse as sparse, numpy as np, pandas as pd
output_dir=os.path.join(os.getcwd(),'output')
A=sparse.load_npz(os.path.join(output_dir,'network_A.npz'))
B=sparse.load_npz(os.path.join(output_dir,'network_B.npz'))

schema=(fg.ModelSchema('SI1I2S')
        .define_compartment(['S','I1','I2'])
        .add_network_layer('layerA')
        .add_network_layer('layerB')
        .add_edge_interaction(name='inf1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
        .add_edge_interaction(name='inf2',from_state='S',to_state='I2',inducer='I2',network_layer='layerB',rate='beta2')
        .add_node_transition(name='rec1',from_state='I1',to_state='S',rate='delta1')
        .add_node_transition(name='rec2',from_state='I2',to_state='S',rate='delta2'))

params={'beta1':0.15,'beta2':0.12,'delta1':1.0,'delta2':1.0}
config=(fg.ModelConfiguration(schema)
        .add_parameter(**params)
        .get_networks(layerA=A,layerB=B))

initial={'percentage':{'I1':5,'I2':5,'S':90}}

sim=fg.Simulation(config,initial_condition=initial,stop_condition={'time':300},nsim=5)
sim.run()
sim.plot_results(show_figure=False,save_figure=True,save_path=os.path.join(output_dir,'results-11.png'))

results=sim.get_results()
time=results[0]; counts=results[1]
res_dict={'time':time}
for idx,comp in enumerate(schema.compartments):
    res_dict[comp]=counts[idx,:]

df=pd.DataFrame(res_dict)
path_csv=os.path.join(output_dir,'results-11.csv')
df.to_csv(path_csv,index=False)

import os, sys, json, numpy as np, scipy.sparse as sparse, pandas as pd, networkx as nx, fastgemf as fg
from scipy.sparse.linalg import eigs

output_dir = os.path.join(os.getcwd(), 'output')
A_csr = sparse.load_npz(os.path.join(output_dir, 'network_A.npz'))
B_csr = sparse.load_npz(os.path.join(output_dir, 'network_B.npz'))

# Model schema
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

# Parameters from previous step (we hardcode values we computed):
beta1 = 0.10401059578111384
beta2 = 0.1765610165080161
delta1 = 1.0
delta2 = 1.0

SIS2_config = (
    fg.ModelConfiguration(SIS2_schema)
    .add_parameter(beta1=beta1, beta2=beta2, delta1=delta1, delta2=delta2)
    .get_networks(layerA=A_csr, layerB=B_csr)
)
print(SIS2_config)

# Initial condition: 90% S, 5% I1, 5% I2 randomly
N = A_csr.shape[0]
state_array = np.zeros(N, dtype=int)  # 0 for S
rand = np.random.RandomState(123)
indices = np.arange(N)
rand.shuffle(indices)
I1_nodes = indices[:int(0.05*N)]
I2_nodes = indices[int(0.05*N):int(0.10*N)]
state_array[I1_nodes] = 1
state_array[I2_nodes] = 2

initial_condition = {'exact': state_array}

sim = fg.Simulation(SIS2_config, initial_condition=initial_condition, stop_condition={'time':500}, nsim=10)
sim.run()
# Save plot
fig_path = os.path.join(output_dir, 'results-11.png')
sim.plot_results(show_figure=False, save_figure=True, save_path=fig_path)

# Get results (last run)
time, state_count, *_ = sim.get_results()
result_dict = {'time': time}
compartments = ['S', 'I1', 'I2']
for idx, comp in enumerate(compartments):
    result_dict[comp] = state_count[idx, :]

results_df = pd.DataFrame(result_dict)
results_csv_path = os.path.join(output_dir, 'results-11.csv')
results_df.to_csv(results_csv_path, index=False)

output = {'fig_path': fig_path, 'csv_path': results_csv_path, 'final': {compartments[i]: state_count[i,-1] for i in range(3)}}

import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os, pandas as pd
from datetime import datetime

# Load networks
A_path=os.path.join(os.getcwd(),'output','network_layer_A.npz')
B_path=os.path.join(os.getcwd(),'output','network_layer_B.npz')
G_A=sparse.load_npz(A_path)
G_B=sparse.load_npz(B_path)

# Define model schema
SIS_comp_schema=(fg.ModelSchema('CompetitiveSIS')
    .define_compartment(['S','I1','I2'])
    .add_network_layer('layerA')
    .add_network_layer('layerB')
    .add_edge_interaction('inf1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
    .add_edge_interaction('inf2',from_state='S',to_state='I2',inducer='I2',network_layer='layerB',rate='beta2')
    .add_node_transition('rec1',from_state='I1',to_state='S',rate='delta1')
    .add_node_transition('rec2',from_state='I2',to_state='S',rate='delta2'))

# Set parameters (delta1=delta2=1)
params={'beta1':0.08,'beta2':0.13,'delta1':1.0,'delta2':1.0}

SIS_instance=(fg.ModelConfiguration(SIS_comp_schema)
    .add_parameter(**params)
    .get_networks(layerA=G_A,layerB=G_B))

print(SIS_instance)

# initial condition 1% each infection
init_cond={'percentage':{'I1':1,'I2':1,'S':98}}

sim=fg.Simulation(SIS_instance,initial_condition=init_cond,stop_condition={'time':300},nsim=10)

sim.run()

# Save plot and csv
output_dir=os.path.join(os.getcwd(),'output')
fig_path=os.path.join(output_dir,'results-11.png')
sim.plot_results(show_figure=False,save_figure=True,save_path=fig_path)

time,state_count,*_=sim.get_results()
compartments=SIS_comp_schema.compartments
results_dict={'time':time}
for idx,comp in enumerate(compartments):
    results_dict[comp]=state_count[idx,:]

df=pd.DataFrame(results_dict)
df.to_csv(os.path.join(output_dir,'results-11.csv'),index=False)

import fastgemf as fg
import numpy as np, os, scipy.sparse as sparse, pandas as pd
current_dir=os.getcwd()
output_dir=os.path.join(current_dir,'output')
GA=sparse.load_npz(os.path.join(output_dir,'layerA.npz'))
GB=sparse.load_npz(os.path.join(output_dir,'layerB.npz'))
# define model schema
schema=(
    fg.ModelSchema('SIS_competitive')
    .define_compartment(['S','A','B'])
    .add_network_layer('layerA')
    .add_network_layer('layerB')
    .add_edge_interaction('adoptA',from_state='S',to_state='A',inducer='A',network_layer='layerA',rate='beta1')
    .add_edge_interaction('adoptB',from_state='S',to_state='B',inducer='B',network_layer='layerB',rate='beta2')
    .add_node_transition('recoverA',from_state='A',to_state='S',rate='delta1')
    .add_node_transition('recoverB',from_state='B',to_state='S',rate='delta2')
)

beta1=0.09
beta2=0.15
delta1=1.0
delta2=1.0

config=(
    fg.ModelConfiguration(schema)
    .add_parameter(beta1=beta1,beta2=beta2,delta1=delta1,delta2=delta2)
    .get_networks(layerA=GA,layerB=GB)
)
print(config)
# initial condition: 5% nodes with A, 5% with B, rest S
N=GA.shape[0]
X0=np.zeros(N,dtype=int) # 0:S
numA=int(0.05*N)
numB=int(0.05*N)
idx=np.random.permutation(N)
A_nodes=idx[:numA]
B_nodes=idx[numA:numA+numB]
X0[A_nodes]=1 # state 1 corresponds to A (index order: S=0,A=1,B=2)
X0[B_nodes]=2
init={'exact':X0}

sim=fg.Simulation(config, initial_condition=init, stop_condition={'time':200}, nsim=3)
sim.run()
# save plot
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(output_dir,'results-11.png'))
# save data from last run
# get compartment counts over time
results=sim.get_results()
# results returns time,state_counts, maybe counts by comp simulation. We capture time and state_count.
time=results[0]
state_counts=results[1]
comp_names=schema.compartments
out={'time':time}
for i,c in enumerate(comp_names):
    out[c]=state_counts[i,:]

df=pd.DataFrame(out)
df.to_csv(os.path.join(output_dir,'results-11.csv'),index=False)

import os, numpy as np, scipy.sparse as sparse, fastgemf as fg, pandas as pd
output_dir = os.path.join(os.getcwd(), 'output')

G_A = sparse.load_npz(os.path.join(output_dir, 'network_A.npz'))
G_B = sparse.load_npz(os.path.join(output_dir, 'network_B.npz'))

# Define model schema
SIS2_schema = (
    fg.ModelSchema('SIS2')
    .define_compartment(['S', 'I1', 'I2'])
    .add_network_layer('layerA')
    .add_network_layer('layerB')
    .add_node_transition(name='rec1', from_state='I1', to_state='S', rate='delta1')
    .add_node_transition(name='rec2', from_state='I2', to_state='S', rate='delta2')
    .add_edge_interaction(name='inf1', from_state='S', to_state='I1', inducer='I1', network_layer='layerA', rate='beta1')
    .add_edge_interaction(name='inf2', from_state='S', to_state='I2', inducer='I2', network_layer='layerB', rate='beta2')
)

# Parameters
params = {
    'beta1': 0.4,
    'delta1': 0.2,
    'beta2': 0.5,
    'delta2': 0.25,
}

SIS2_config = (
    fg.ModelConfiguration(SIS2_schema)
    .add_parameter(**params)
    .get_networks(layerA=G_A, layerB=G_B)
)

print(SIS2_config)

# Initial condition 1: 1% I1, 1% I2
n = G_A.shape[0]
I1_seed = np.random.choice(n, size=int(0.01*n), replace=False)
remaining = list(set(range(n)) - set(I1_seed))
I2_seed = np.random.choice(remaining, size=int(0.01*n), replace=False)
X0 = np.zeros(n, dtype=int)  # S=0, I1=1, I2=2 per index mapping of compartments automatically maybe assigned 0,1,2
# Determine mapping order as defined in compartments list
# Assume S=0, I1=1, I2=2
X0[I1_seed] = 1
X0[I2_seed] = 2

initial_condition = {'exact': X0}

sim = fg.Simulation(SIS2_config, initial_condition=initial_condition, stop_condition={'time':300}, nsim=10)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(output_dir, 'results-11.png'))

time, state_count, *_ = sim.get_results()

results = {'time': time}
for idx, comp in enumerate(SIS2_schema.compartments):
    results[comp] = state_count[idx,:]

df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, 'results-11.csv'), index=False)

import os, scipy.sparse as sparse, numpy as np, fastgemf as fg, pandas as pd, networkx as nx

# Load networks
A_path=os.path.join(os.getcwd(),'output','layer_A.npz')
B_path=os.path.join(os.getcwd(),'output','layer_B.npz')
A=sparse.load_npz(A_path)
B=sparse.load_npz(B_path)

# Model schema SI1I2S
model_schema=(fg.ModelSchema('SI1I2S')
    .define_compartment(['S','I1','I2'])
    .add_network_layer('layerA')
    .add_network_layer('layerB')
    .add_edge_interaction('infection1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
    .add_edge_interaction('infection2',from_state='S',to_state='I2',inducer='I2',network_layer='layerB',rate='beta2')
    .add_node_transition('rec1',from_state='I1',to_state='S',rate='delta1')
    .add_node_transition('rec2',from_state='I2',to_state='S',rate='delta2')
)

beta1=0.10401059578111396
beta2=0.13639308317445792
delta1=1.0
delta2=1.0

model_config=(fg.ModelConfiguration(model_schema)
    .add_parameter(beta1=beta1,beta2=beta2,delta1=delta1,delta2=delta2)
    .get_networks(layerA=A,layerB=B)
)

# initial condition 5% infected each
n=A.shape[0]
I1_nodes=np.random.choice(n, size=int(0.05*n), replace=False)
remaining=list(set(range(n))-set(I1_nodes))
I2_nodes=np.random.choice(remaining,size=int(0.05*n),replace=False)
X0=np.zeros(n,dtype=int)
X0[I1_nodes]=1  # I1 state index 1
X0[I2_nodes]=2  # I2 state index 2
initial_condition={'exact':X0}

sim=fg.Simulation(model_config,initial_condition=initial_condition,stop_condition={'time':300},nsim=1)

sim.run()

sim.plot_results(show_figure=False,save_figure=True,save_path=os.path.join(os.getcwd(),'output','results-11.png'))

time, state_count, *_=sim.get_results()
results={'time':time}
for i,comp in enumerate(model_schema.compartments):
    results[comp]=state_count[i,:]

df=pd.DataFrame(results)
df.to_csv(os.path.join(os.getcwd(),'output','results-11.csv'),index=False)
