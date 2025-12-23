
import math, numpy as np
z=3
R0=4
T=R0/4 # =1
# compute probability mass function for k up to say 20 for Poisson 3
pk=[math.exp(-z)*z**k/math.factorial(k) for k in range(21)]
# compute initial metrics
mean_k=sum(k*pk[k] for k in range(len(pk)))
q=sum(k*(k-1)*pk[k] for k in range(len(pk)))/mean_k
print(mean_k,q)
# now targeted removal: vaccinate fraction x of k=10 nodes
pk_new=list(pk)
def R0_eff(x):
    pk2=pk.copy()
    pk2[10]=pk[10]*(1-x)
    # remaining fraction
    f=x*pk[10]
    remaining=1-f
    mean_k_new=sum(k*pk2[k] for k in range(len(pk2)))/remaining
    q_new=sum(k*(k-1)*pk2[k] for k in range(len(pk2)))/remaining/mean_k_new
    return T*q_new
# find x such that R0_eff<1
xs=np.linspace(0,1,101)
for x in xs:
    if R0_eff(x)<1:
        print('threshold',x)
        break

import math, numpy as np
z=3
# compute Poisson distribution upto 30
maxk=30
pk=[math.exp(-z)*z**k/math.factorial(k) for k in range(maxk+1)]
mean_k=sum(k*pk[k] for k in range(maxk+1))
q=sum(k*(k-1)*pk[k] for k in range(maxk+1))/mean_k
print('mean',mean_k,'q',q)
T=1 # R0=4 implies T q =4 => T=4/q. Wait q=4, so T=1.
print('T',T)

def R0_eff(x):
    # remove fraction x of k=10 nodes
    pk2=pk.copy()
    pk2[10]=pk[10]*(1-x)
    remaining=1 - x*pk[10]
    # renormalize
    pk2=[p/remaining for p in pk2]
    mean=sum(k*pk2[k] for k in range(maxk+1))
    q_val=sum(k*(k-1)*pk2[k] for k in range(maxk+1))/mean
    return T*q_val
# find x threshold
thr=None
for x in np.linspace(0,1,1001):
    if R0_eff(x)<1:
        thr=x
        break
print('threshold',thr)

import os, networkx as nx, numpy as np, scipy.sparse as sparse
import random, math
current_dir=os.getcwd()
output_dir=os.path.join(current_dir,'output')
os.makedirs(output_dir,exist_ok=True)
# parameters
n=5000
r=3
p=0.5
# sample degrees
np.random.seed(42)
while True:
    degs=np.random.negative_binomial(r, p, size=n)
    # ensure positive degree maybe keep zeros but later disclaim ensures connectedness maybe not.
    if sum(degs)%2==0 and np.all(degs>=0):
        break
G=nx.configuration_model(degs, create_using=nx.Graph())
G.remove_edges_from(nx.selfloop_edges(G))
# compute mean, k2
degrees=[d for d in dict(G.degree()).values()]
mean_k=np.mean(degrees)
second_moment=np.mean(np.array(degrees)**2)
q=(second_moment-mean_k)/mean_k
print('mean',mean_k,'q',q)
# save network
sparse.save_npz(os.path.join(output_dir,'network.npz'), nx.to_scipy_sparse_array(G))
import math
z=3
P10=math.exp(-z)*z**10/math.factorial(10)
P10
import importlib, sys, subprocess, json, os, math, random, numpy as np
import math
z=3
maxk=30
pk=[math.exp(-z)*z**k/math.factorial(k) for k in range(maxk+1)]
remaining=[p for p in pk]
remaining[10]=0
rem_total=sum(remaining)
remaining=[p/rem_total for p in remaining]
mean=sum(k*remaining[k] for k in range(maxk+1))
q=sum(k*(k-1)*remaining[k] for k in range(maxk+1))/mean
R0_new=q # T=1
R0_new
import importlib, sys, subprocess, json, os
try:
    import fastgemf as fg
    print('fastgemf found',fg.__version__)
except ImportError as e:
    print('not found')
import os, random, math, numpy as np, networkx as nx, scipy.sparse as sparse, pandas as pd
from collections import deque

current_dir=os.getcwd()
output_dir=os.path.join(current_dir,'output')
os.makedirs(output_dir, exist_ok=True)

# 1. Generate network
n=10000
z=3
"""Generate Poisson degree distribution via configuration model"""
# sample degree sequence
deg_seq=np.random.poisson(lam=z, size=n)
# ensure even sum
if deg_seq.sum()%2==1:
    idx=np.random.randint(0,n)
    deg_seq[idx]+=1
G=nx.configuration_model(deg_seq, create_using=nx.Graph())
G.remove_edges_from(nx.selfloop_edges(G))

sparse.save_npz(os.path.join(output_dir,'network.npz'), nx.to_scipy_sparse_array(G))

# compute q
deg=np.array([d for _,d in G.degree()])
mean_k=deg.mean()
second_moment=(deg**2).mean()
q=(second_moment-mean_k)/mean_k
print('Network mean',mean_k,'q',q)

# epidemic parameters
beta=1.0   # per contact infection rate
gamma=1.0  # recovery rate

# simulation function
def run_SIR(G, infected_init, vaccinated_set):
    # states: 0=S,1=I,2=R,3=V
    n=G.number_of_nodes()
    state=np.zeros(n, dtype=int)
    state[list(vaccinated_set)] = 3
    state[list(infected_init)] = 1
    t=0
    time=[0]
    S=[np.sum(state==0)]
    I=[len(infected_init)]
    R=[0]
    while np.any(state==1):
        # Gillespie-like step: iterate over infected list compute all rates
        infection_events=[]
        total_rate=0
        infected_nodes=np.where(state==1)[0]
        # infection edges
        for i in infected_nodes:
            for nbr in G.neighbors(i):
                if state[nbr]==0:
                    total_rate+=beta
                    infection_events.append((beta,i,nbr))
        # recovery events
        for i in infected_nodes:
            total_rate+=gamma
            infection_events.append((gamma,i,None))
        if total_rate==0:
            break
        # time increment
        dt=np.random.exponential(1/total_rate)
        t+=dt
        # choose event
        r=np.random.random()*total_rate
        cumulative=0
        for rate,src,target in infection_events:
            cumulative+=rate
            if r<=cumulative:
                if target is None:
                    # recovery
                    state[src]=2
                else:
                    # infection
                    state[target]=1
                break
        time.append(t)
        S.append(np.sum(state==0))
        I.append(np.sum(state==1))
        R.append(np.sum(state==2))
    return pd.DataFrame({'time':time,'S':S,'I':I,'R':R})

# Scenario 1: random vaccination f=0.75
f=0.75
vaccinated=set(random.sample(range(n), int(f*n)))
infected_init=set(random.sample(list(set(range(n))-vaccinated), 10))
result1=run_SIR(G, infected_init, vaccinated)
result1.to_csv(os.path.join(output_dir,'results-11.csv'), index=False)
print('Scenario1 final infected', result1[['R']].iloc[-1].values[0])

# Scenario2: vaccinate all nodes with degree ==10
vaccinated2=set([node for node,deg in G.degree() if deg==10])
f2=len(vaccinated2)/n
print('fraction vacc degree10',f2)
infected_init2=set(random.sample(list(set(range(n))-vaccinated2), 10))
result2=run_SIR(G, infected_init2, vaccinated2)
result2.to_csv(os.path.join(output_dir,'results-12.csv'), index=False)
print('Scenario2 final infected', result2[['R']].iloc[-1].values[0])
import pandas as pd, os, numpy as np
output_dir=os.path.join(os.getcwd(),'output')
r1=pd.read_csv(os.path.join(output_dir,'results-11.csv'))
r2=pd.read_csv(os.path.join(output_dir,'results-12.csv'))
final1=r1[['R']].iloc[-1].values[0]
final2=r2[['R']].iloc[-1].values[0]
peakI1=r1['I'].max()
peakI2=r2['I'].max()
duration1=r1['time'].iloc[-1]
duration2=r2['time'].iloc[-1]

import math, numpy as np
z=3
maxk=30
pk=[math.exp(-z)*z**k/math.factorial(k) for k in range(maxk+1)]
T=1
mean=sum(k*pk[k] for k in range(maxk+1))
q=sum(k*(k-1)*pk[k] for k in range(maxk+1))/mean
print('q baseline',q)

def q_after(x):
    pk2=pk.copy()
    pk2[10]=pk[10]*(1-x)
    denom=1-x*pk[10]
    pk2=[p/denom for p in pk2]
    mean2=sum(k*pk2[k] for k in range(maxk+1))
    q2=sum(k*(k-1)*pk2[k] for k in range(maxk+1))/mean2
    return q2

for x in [0,0.5,1]:
    print(x,q_after(x))

import os, random, math, numpy as np, networkx as nx, scipy.sparse as sparse, pandas as pd
from tqdm import trange

current_dir=os.getcwd()
output_dir=os.path.join(current_dir,'output')
os.makedirs(output_dir, exist_ok=True)
# parameters
n=15000
z=3
np.random.seed(0)
# generate Poisson network
while True:
    deg_seq=np.random.poisson(lam=z, size=n)
    if deg_seq.sum()%2==0:
        break
G=nx.configuration_model(deg_seq, create_using=nx.Graph())
G.remove_edges_from(nx.selfloop_edges(G))
# compute q
deg_arr=np.array([d for _,d in G.degree()])
mean_k=deg_arr.mean()
second= (deg_arr**2).mean()
q=(second-mean_k)/mean_k
print('Network mean',mean_k,'q',q)
# disease parameters
gamma=1.0
beta=4.0*gamma/q # to achieve R0=4 theoretically
print('beta',beta)

# simulation

def run_SIR(beta,gamma,G,infected_init,vaccinated_set):
    n=G.number_of_nodes()
    # states 0=S 1=I 2=R 3=V
    state=np.zeros(n, dtype=np.int8)
    state[list(vaccinated_set)] = 3
    susceptibles=set(np.where(state==0)[0])
    infected=set(infected_init)
    susceptibles-=infected
    for node in infected:
        state[node]=1
    t=0.0
    rec_times={}
    # schedule recovery times
    for i in infected:
        rec_times[i]= t + np.random.exponential(1/gamma)
    # event-driven simulation using queue of infection events computed on the fly (approx) - simpler: time-step small dt.
    dt=0.1
    times=[0.0]
    I_counts=[len(infected)]
    S_counts=[len(susceptibles)]
    R_counts=[0]
    while infected:
        t+=dt
        new_infected=set()
        # infection attempts
        for i in list(infected):
            for nbr in G.neighbors(i):
                if nbr in susceptibles:
                    if random.random()<1-math.exp(-beta*dt):
                        new_infected.add(nbr)
        for node in new_infected:
            infected.add(node)
            susceptibles.remove(node)
            rec_times[node]=t + np.random.exponential(1/gamma)
            state[node]=1
        # recoveries
        for node in list(infected):
            if t>=rec_times[node]:
                infected.remove(node)
                state[node]=2
        times.append(t)
        I_counts.append(len(infected))
        S_counts.append(len(susceptibles))
        R_counts.append(np.sum(state==2))
        if len(times)>5000:
            break
    return pd.DataFrame({'time':times,'S':S_counts,'I':I_counts,'R':R_counts})

# Scenario random vaccination 0.75
f_random=0.75
runs=20
results_list=[]
for run in trange(runs):
    vaccinated=set(random.sample(range(n), int(f_random*n)))
    possible=list(set(range(n))-vaccinated)
    infected_init=random.sample(possible, 10)
    df=run_SIR(beta,gamma,G, infected_init, vaccinated)
    attack=df['R'].iloc[-1]
    peakI=max(df['I'])
    duration=df['time'].iloc[-1]
    results_list.append({'run':run,'attack':attack,'peakI':peakI,'duration':duration})
agg1=pd.DataFrame(results_list)
agg1.to_csv(os.path.join(output_dir,'results-21.csv'), index=False)

# Scenario degree 10 vaccination
deg10_nodes=[node for node,d in G.degree() if d==10]
vaccinated2=set(deg10_nodes)
frac2=len(vaccinated2)/n
print('fraction vacc 10',frac2)
results2=[]
for run in trange(runs):
    possible=list(set(range(n))-vaccinated2)
    infected_init=random.sample(possible, 10)
    df=run_SIR(beta,gamma,G, infected_init, vaccinated2)
    attack=df['R'].iloc[-1]
    peakI=max(df['I'])
    duration=df['time'].iloc[-1]
    results2.append({'run':run,'attack':attack,'peakI':peakI,'duration':duration})
agg2=pd.DataFrame(results2)
agg2.to_csv(os.path.join(output_dir,'results-22.csv'), index=False)
print(agg1.describe())
print(agg2.describe())
import math, numpy as np
p=0.25
P=[p*(1-p)**k for k in range(30)]
P10=P[10]
print('P10',P10)
# compute moments
k_arr=np.arange(len(P))
Pk=np.array(P)
z=(k_arr*Pk).sum()
kk1=(k_arr*(k_arr-1)*Pk).sum()
print('z',z,'kk1',kk1)
mean_excess=kk1/z
print('q',mean_excess)

def compute_q(f):
    total_removed=f*P10
    if total_removed>=1: return 0
    z_prime=(z - f*10*P10)/(1-total_removed)
    kk1_prime=(kk1 - f*90*P10)/(1-total_removed)
    q_prime=kk1_prime/z_prime
    return q_prime

for f in [0,0.5,1]:
    print(f,compute_q(f))
import os, random, math, sys, json, itertools, collections, statistics, functools
import math, numpy as np
p=0.25
P=[p*(1-p)**k for k in range(50)]
Pk=np.array(P)

k_vals=np.arange(len(Pk))

z=(k_vals*Pk).sum()
kk1=(k_vals*(k_vals-1)*Pk).sum()
q=kk1/z
print('z',z,q)
P10=Pk[10]
print('P10',P10)
# define function q_prime(f)
def q_prime(f):
    removed= f*P10
    if removed>=1: return 0
    z_prime=(z - f*10*P10)/(1-removed)
    kk1_prime=(kk1 - f*10*9*P10)/(1-removed)
    return kk1_prime/z_prime

# find f such that q_prime(f)=1
from bisect import bisect_left
F=np.linspace(0,1,10001)
qs=[q_prime(f) for f in F]
idx=bisect_left(qs[::-1],1) # reversed cause decreasing
f_req=F[len(F)-1-idx]
print('f_req',f_req)
print('q_new',q_prime(f_req))
import math, numpy as np
# negative binomial r=3, p=0.5
r=3
p=0.5
def negbin_pmf(k, r, p):
    from math import comb
    return comb(k+r-1, k)*(1-p)**k * p**r

K=np.arange(0,50)
P=np.array([negbin_pmf(k,r,p) for k in K])
print(P.sum())
mean=(K*P).sum()
second=(K**2*P).sum()
print('mean',mean,'second',second)
q=(second - mean)/mean
print('q',q)
# compute removal of k=10 nodes
P10=P[10]
f=P10
mean_new=( (K*P).sum() - 10*P10)/(1 - P10)
second_new=(( (K**2)*P).sum() - 100*P10)/(1-P10)
q_new=(second_new - mean_new)/mean_new
print('q_new',q_new)

import pandas as pd, os, numpy as np
output_dir = os.path.join(os.getcwd(), 'output')
res11 = pd.read_csv(os.path.join(output_dir, 'results-11.csv'))
res12 = pd.read_csv(os.path.join(output_dir, 'results-12.csv'))

# Compute metrics function
def metrics(df):
    peak_I1 = df['I1'].max() / 1000
    peak_I2 = df['I2'].max() / 1000
    final_I1 = df['I1'].iloc[-1]/1000
    final_I2 = df['I2'].iloc[-1]/1000
    peak_time_I1 = df.loc[df['I1'].idxmax(), 'time']
    peak_time_I2 = df.loc[df['I2'].idxmax(), 'time']
    return {'peak_I1': peak_I1, 'peak_I2': peak_I2, 'final_I1': final_I1, 'final_I2': final_I2,
            'peak_time_I1': peak_time_I1, 'peak_time_I2': peak_time_I2}

m11 = metrics(res11)
m12 = metrics(res12)

import os, numpy as np, scipy.sparse as sparse, pandas as pd, fastgemf as fg
output_dir = os.path.join(os.getcwd(), 'output')
A_csr = sparse.load_npz(os.path.join(output_dir, 'network_A.npz'))
B_csr = sparse.load_npz(os.path.join(output_dir, 'network_B.npz'))

beta1 = 0.14
beta2 = 0.14
delta1 = delta2 = 1.0
schema = (fg.ModelSchema('CompetitiveSIS')
          .define_compartment(['S','I1','I2'])
          .add_network_layer('A')
          .add_network_layer('B')
          .add_edge_interaction('inf1', from_state='S', to_state='I1', inducer='I1', network_layer='A', rate='beta1')
          .add_edge_interaction('inf2', from_state='S', to_state='I2', inducer='I2', network_layer='B', rate='beta2')
          .add_node_transition('rec1', from_state='I1', to_state='S', rate='delta1')
          .add_node_transition('rec2', from_state='I2', to_state='S', rate='delta2'))
config = (fg.ModelConfiguration(schema)
          .add_parameter(beta1=beta1, beta2=beta2, delta1=delta1, delta2=delta2)
          .get_networks(A=A_csr, B=B_csr))
N = A_csr.shape[0]
state = np.zeros(N, dtype=int)
rand = np.random.RandomState(99)
perm = rand.permutation(N)
I1_nodes = perm[:int(0.05*N)]
I2_nodes = perm[int(0.05*N):int(0.1*N)]
state[I1_nodes]=1
state[I2_nodes]=2
init={'exact': state}
sim=fg.Simulation(config, initial_condition=init, stop_condition={'time':600}, nsim=5)
sim.run()
fig_path=os.path.join(output_dir,'results-13.png')
sim.plot_results(show_figure=False, save_figure=True, save_path=fig_path)
time, state_count, *_ = sim.get_results()
res={'time':time,'S':state_count[0],'I1':state_count[1],'I2':state_count[2]}
import pandas as pd
csv_path=os.path.join(output_dir,'results-13.csv')
pd.DataFrame(res).to_csv(csv_path,index=False)
final={'S':state_count[0,-1],'I1':state_count[1,-1],'I2':state_count[2,-1]}

import os, pandas as pd
output_dir = os.path.join(os.getcwd(), 'output')
res24 = pd.read_csv(os.path.join(output_dir,'results-24.csv'))

def metrics(df):
    N=1000
    peak_I1=df['I1'].max()/N
    peak_I2=df['I2'].max()/N
    final_I1=df['I1'].iloc[-1]/N
    final_I2=df['I2'].iloc[-1]/N
    return {'peak_I1':peak_I1,'peak_I2':peak_I2,'final_I1':final_I1,'final_I2':final_I2}

m24=metrics(res24)

import os, networkx as nx, numpy as np, scipy.sparse as sparse, scipy.sparse.linalg as spla
from pathlib import Path

N=1000
seed=42
np.random.seed(seed)

out_dir=os.path.join(os.getcwd(),'output'); Path(out_dir).mkdir(exist_ok=True)

G_A=nx.barabasi_albert_graph(N,3,seed=seed)
G_B=nx.watts_strogatz_graph(N,6,0.1,seed=seed)

A_csr=nx.to_scipy_sparse_array(G_A, dtype=float, format='csr')
B_csr=nx.to_scipy_sparse_array(G_B, dtype=float, format='csr')

lambda1_A=float(spla.eigs(A_csr, k=1, which='LR', return_eigenvectors=False)[0].real)
lambda1_B=float(spla.eigs(B_csr, k=1, which='LR', return_eigenvectors=False)[0].real)

lambda1_A, lambda1_B
import networkx as nx, numpy as np, scipy.sparse as sparse, scipy.sparse.linalg as spla, os
from pathlib import Path
N=1000
seed=42
np.random.seed(seed)
out_dir=os.path.join(os.getcwd(),'output'); Path(out_dir).mkdir(exist_ok=True)

# Load or create networks
import pickle, sys
A_csr=sparse.load_npz(os.path.join(out_dir,'network_layerA.npz'))
B_csr=sparse.load_npz(os.path.join(out_dir,'network_layerB.npz'))

# Compute dominant eigenvectors
val_A, vec_A=spla.eigs(A_csr, k=1, which='LR')
val_B, vec_B=spla.eigs(B_csr, k=1, which='LR')
vec_A=vec_A[:,0].real
vec_B=vec_B[:,0].real

# Normalize
vec_A/=np.linalg.norm(vec_A)
vec_B/=np.linalg.norm(vec_B)

cosine=float(np.dot(vec_A, vec_B))

import pandas as pd, os, numpy as np, matplotlib.pyplot as plt
output_dir=os.path.join(os.getcwd(),'output')
df=pd.read_csv(os.path.join(output_dir,'results-11.csv'))
print(df.tail())
last=df.iloc[-1]
print('Final proportions A', last['A']/1000, 'B', last['B']/1000)
peakA=df['A'].max()/1000
peakB=df['B'].max()/1000
print('Peak A', peakA,' Peak B', peakB)

import fastgemf as fg, numpy as np, os, scipy.sparse as sparse, pandas as pd
from scipy.sparse.linalg import eigs
import networkx as nx
current_dir=os.getcwd()
output_dir=os.path.join(current_dir,'output')
GA=sparse.load_npz(os.path.join(output_dir,'layerA.npz'))
GB=sparse.load_npz(os.path.join(output_dir,'layerB.npz'))
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

delta1=1.0
delta2=1.0

# parameter sweep
beta1_list=[0.07,0.09,0.11]  # around threshold
beta2_list=[0.14,0.15,0.18]

results_summary=[]
run_id=20
for b1 in beta1_list:
  for b2 in beta2_list:
    config=(
        fg.ModelConfiguration(schema)
        .add_parameter(beta1=b1,beta2=b2,delta1=delta1,delta2=delta2)
        .get_networks(layerA=GA,layerB=GB)
    )
    N=GA.shape[0]
    X0=np.zeros(N,dtype=int)
    idx=np.random.permutation(N)
    X0[idx[:int(0.05*N)]]=1
    X0[idx[int(0.05*N):int(0.10*N)]]=2
    sim=fg.Simulation(config, initial_condition={'exact':X0}, stop_condition={'time':200}, nsim=1)
    sim.run()
    time, counts=sim.get_results()[:2]
    finalA=counts[1,-1]/N
    finalB=counts[2,-1]/N
    peakA=counts[1,:].max()/N
    peakB=counts[2,:].max()/N
    results_summary.append({'beta1':b1,'beta2':b2,'finalA':finalA,'finalB':finalB,'peakA':peakA,'peakB':peakB})
    # Save each run file
    path_csv=os.path.join(output_dir,f'results-{run_id}.csv')
    df=pd.DataFrame({'time':time,'S':counts[0,:],'A':counts[1,:],'B':counts[2,:]})
    df.to_csv(path_csv,index=False)
    run_id+=1
# save summary
pd.DataFrame(results_summary).to_csv(os.path.join(output_dir,'summary.csv'),index=False)

import pandas as pd, os, numpy as np
output_dir = os.path.join(os.getcwd(), 'output')
df = pd.read_csv(os.path.join(output_dir, 'results-11.csv'))
# Compute prevalence fractions
pop = 1000
# Get last timestep values
last = df.iloc[-1]
I1_end = last['I1']/pop
I2_end = last['I2']/pop

# Peak infection
peak_I1 = df['I1'].max()/pop
peak_I2 = df['I2'].max()/pop

results_summary = {
    'I1_final_fraction': I1_end,
    'I2_final_fraction': I2_end,
    'I1_peak': peak_I1,
    'I2_peak': peak_I2
}
