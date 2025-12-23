
import fastgemf as fg, numpy as np, pandas as pd, scipy.sparse as sparse, os, math, random, networkx as nx
from collections import Counter

output_dir = os.path.join(os.getcwd(), 'output')
G_csr = sparse.load_npz(os.path.join(output_dir,'network.npz'))
N = G_csr.shape[0]

# Model schema
SIR_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S','I','R'])
    .add_network_layer('contact')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact', rate='beta')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
)

# parameter select
mean_k = (G_csr.sum(axis=1)).mean()
k2 = ((G_csr.sum(axis=1)**2)).mean()
q_val = (k2 - mean_k)/mean_k
R0 = 4.0
beta = R0 / q_val  # gamma=1
params = {'beta': float(beta), 'gamma': 1.0}

# set model configuration
schema_instance = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(**params)
    .get_networks(contact=G_csr)
)

print('beta', beta, 'q', q_val, 'mean_k', mean_k)

# Function to create initial condition given vaccination set

def create_ic(vaccinated_indices, initial_infected_indices):
    X0 = np.zeros(N, dtype=int)  # default S (0)
    X0[vaccinated_indices] = 2  # R
    X0[initial_infected_indices] = 1  # I overrides if also vaccinated? but we ensure non vaccinated.
    return {'exact': X0}

rng = np.random.default_rng(123)

# scenario 0: no vaccination baseline
initial_infected = rng.choice(N, 5, replace=False)  # 5 seeds infected
ic0 = create_ic([], initial_infected)

# scenario 1: random vaccination 75%
vac1 = rng.choice(N, int(0.75*N), replace=False)
remaining = np.setdiff1d(np.arange(N), vac1)
init_inf1 = rng.choice(remaining, 5, replace=False)
ic1 = create_ic(vac1, init_inf1)

# scenario 2: targeted degree10 vaccination (all)
# need indices of degree 10 nodes
G_nx = nx.from_scipy_sparse_array(G_csr)
deg_list = dict(G_nx.degree())
deg10_nodes = np.array([n for n,d in deg_list.items() if d==10])
remaining2 = np.setdiff1d(np.arange(N), deg10_nodes)
init_inf2 = rng.choice(remaining2, 5, replace=False)
ic2 = create_ic(deg10_nodes, init_inf2)

# scenario 3: targeted degree10 vacc 8% of population (~0.8 of critical) for contrast
num_vac3 = int(0.08*N)
if num_vac3 > len(deg10_nodes):
    deg10_subset = deg10_nodes
else:
    deg10_subset = rng.choice(deg10_nodes, num_vac3, replace=False)
remaining3 = np.setdiff1d(np.arange(N), deg10_subset)
init_inf3 = rng.choice(remaining3,5,replace=False)
ic3 = create_ic(deg10_subset, init_inf3)

sims = []
conditions = [ic0, ic1, ic2, ic3]
labels = ['baseline','random75','targeted_deg10_all','targeted_deg10_8']
for idx, (ic, label) in enumerate(zip(conditions, labels)):
    sim = fg.Simulation(schema_instance, initial_condition=ic, stop_condition={'time':100}, nsim=30)
    sim.run()
    time, state_count, *_ = sim.get_results()
    df = pd.DataFrame({'time': time})
    for i, comp in enumerate(['S','I','R']):
        df[comp] = state_count[i,:]
    csv_path = os.path.join(output_dir, f'results-1{idx}.csv')
    df.to_csv(csv_path, index=False)
    # save figure
    png_path = os.path.join(output_dir, f'results-1{idx}.png')
    sim.plot_results(show_figure=False, save_figure=True, save_path=png_path)
    sims.append({'label':label,'csv':csv_path,'png':png_path})

result_paths = sims

import os, numpy as np, networkx as nx, scipy.sparse as sparse, fastgemf as fg, random, pandas as pd
# Load network
csr=sparse.load_npz(os.path.join(os.getcwd(),'output','network.npz'))
N=csr.shape[0]
# Build SIR schema
schema=(
    fg.ModelSchema('SIR')
    .define_compartment(['S','I','R'])
    .add_network_layer('contact')
    .add_node_transition(name='recover', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infect', from_state='S', to_state='I', inducer='I', network_layer='contact', rate='beta')
)
# Parameters
beta=1.01
gamma=1.0
config=(
    fg.ModelConfiguration(schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact=csr)
)
# Scenario 1: random vaccination 75%
rand_mask=np.random.rand(N) < 0.75
initial_state=np.zeros(N, dtype=int) # S=0
initial_state[rand_mask]=2 # R
# seed infections in remaining susceptible nodes
susceptible_indices=np.where(initial_state==0)[0]
seed=np.random.choice(susceptible_indices, size=10, replace=False)
initial_state[seed]=1
sim1=fg.Simulation(config, initial_condition={'exact': initial_state}, stop_condition={'time':100}, nsim=1)
sim1.run()
time1, state_count1, *_=sim1.get_results()
res1={'time':time1}
for idx,comp in enumerate(schema.compartments):
    res1[comp]=state_count1[idx,:]
res1_df=pd.DataFrame(res1)
os.makedirs(os.path.join(os.getcwd(),'output'), exist_ok=True)
res1_df.to_csv(os.path.join(os.getcwd(),'output','results-11.csv'), index=False)
sim1.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-11.png'))
# Scenario 2: vaccinate all degree 10 nodes
# Identify degree10 nodes from csr degrees
degrees=np.array(csr.sum(axis=1)).flatten()
deg10_mask=degrees==10
initial_state2=np.zeros(N, dtype=int)
initial_state2[deg10_mask]=2
sus2=np.where(initial_state2==0)[0]
seed2=np.random.choice(sus2, size=10, replace=False)
initial_state2[seed2]=1
sim2=fg.Simulation(config, initial_condition={'exact': initial_state2}, stop_condition={'time':100}, nsim=1)
sim2.run()
time2, state_count2,*_=sim2.get_results()
res2={'time':time2}
for idx,comp in enumerate(schema.compartments):
    res2[comp]=state_count2[idx,:]
res2_df=pd.DataFrame(res2)
res2_df.to_csv(os.path.join(os.getcwd(),'output','results-12.csv'), index=False)
sim2.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(),'output','results-12.png'))

import numpy as np, networkx as nx, scipy.sparse as sparse, random, os, math, tqdm, pickle
from collections import deque

# Load network
net_path=os.path.join(os.getcwd(),'output','network.npz')
A=sparse.load_npz(net_path)
G=nx.from_scipy_sparse_array(A)
N=G.number_of_nodes()

# parameters
period=4  # infectious duration steps
p_trans=1/3  # per contact per step infection probability


def run_sim(vaccinated_nodes, initial_infected, max_steps=100):
    S=set(range(N))
    I=set()
    R=set(vaccinated_nodes)
    S-=R
    S-=set(initial_infected)
    I|=set(initial_infected)
    timeline=[(0, len(S), len(I), len(R))]
    infect_time={i:0 for i in I}
    for t in range(1,max_steps+1):
        new_inf=[]
        new_rec=[]
        for i in list(I):
            # transmit to neighbors
            for nbr in G.neighbors(i):
                if nbr in S and random.random()<p_trans:
                    new_inf.append(nbr)
            infect_time[i]+=1
            if infect_time[i]>=period:
                new_rec.append(i)
        for n in new_inf:
            S.discard(n)
            I.add(n)
            infect_time[n]=0
        for n in new_rec:
            I.discard(n)
            R.add(n)
            infect_time.pop(n,None)
        timeline.append((t, len(S), len(I), len(R)))
        if len(I)==0:
            break
    return np.array(timeline)

scenarios=[]
# baseline no vaccination
vacc0=set()
initial_infected=random.sample(range(N), int(0.01*N))
base=run_sim(vacc0, initial_infected)
scenarios.append(('baseline', base))

# random vaccination 75%
random_vacc=set(random.sample(range(N), int(0.75*N)))
S_nodes=list(set(range(N))-random_vacc)
init_inf=random.sample(S_nodes, max(1,int(0.01*N)))
rand75=run_sim(random_vacc, init_inf)
scenarios.append(('random75', rand75))

# degree 10 vaccination
deg10_nodes=[n for n,d in G.degree() if d==10]
init_inf=random.sample(list(set(range(N))-set(deg10_nodes)), max(1,int(0.01*N)))
deg10=run_sim(set(deg10_nodes), init_inf)
scenarios.append(('deg10', deg10))

# save results
out_dir=os.path.join(os.getcwd(),'output')
import pandas as pd, pathlib
pathlib.Path(out_dir).mkdir(exist_ok=True)
for idx,(name,data) in enumerate(scenarios):
    df=pd.DataFrame(data, columns=['time','S','I','R'])
    df.to_csv(os.path.join(out_dir, f'results-1{idx}.csv'), index=False)
print('saved scenario CSVs')
