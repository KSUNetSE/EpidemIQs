
import networkx as nx
import numpy as np
import os
import scipy.sparse as sparse
from collections import defaultdict

# Parameters for networks
n = 1000
mean_k = 8  # target mean degree a bit higher for epidemic spread
p_er = mean_k / (n-1)

# Create Erdős-Rényi network (homogeneous)
G_er = nx.erdos_renyi_graph(n, p_er, seed=42)

# Create Barabási-Albert network (heterogeneous)
m_ba = mean_k // 2  # each new node attaches to m existing -> mean degree ~2m
G_ba = nx.barabasi_albert_graph(n, m_ba, seed=42)

# Save networks
output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

sparse.save_npz(os.path.join(output_dir, 'network_er.npz'), nx.to_scipy_sparse_array(G_er))
sparse.save_npz(os.path.join(output_dir, 'network_ba.npz'), nx.to_scipy_sparse_array(G_ba))

# compute degree stats
k_er = np.array([d for _, d in G_er.degree()])
mean_k_er = k_er.mean()
second_moment_er = (k_er**2).mean()
q_er = (second_moment_er - mean_k_er)/mean_k_er

k_ba = np.array([d for _, d in G_ba.degree()])
mean_k_ba = k_ba.mean()
second_moment_ba = (k_ba**2).mean()
q_ba = (second_moment_ba - mean_k_ba)/mean_k_ba

stats = {
    'mean_k_er': mean_k_er,
    'second_moment_er': second_moment_er,
    'q_er': q_er,
    'mean_k_ba': mean_k_ba,
    'second_moment_ba': second_moment_ba,
    'q_ba': q_ba,
}

stats
mean_k_er = 8.036
beta = (2.5/7)/mean_k_er
beta
import fastgemf as fg
import scipy.sparse as sparse
import networkx as nx
import numpy as np
import os, json
from scipy.integrate import solve_ivp
import pandas as pd

output_dir = os.path.join(os.getcwd(), 'output')

# Load networks
er_csr = sparse.load_npz(os.path.join(output_dir, 'network_er.npz'))
ba_csr = sparse.load_npz(os.path.join(output_dir, 'network_ba.npz'))

# Degree stats
G_er = nx.from_scipy_sparse_array(er_csr)
G_ba = nx.from_scipy_sparse_array(ba_csr)

k_er = np.array([d for _, d in G_er.degree()]); mean_k_er = k_er.mean(); second_moment_er = (k_er**2).mean(); q_er = (second_moment_er - mean_k_er)/mean_k_er
k_ba = np.array([d for _, d in G_ba.degree()]); mean_k_ba = k_ba.mean(); second_moment_ba = (k_ba**2).mean(); q_ba = (second_moment_ba - mean_k_ba)/mean_k_ba

R0_target = 2.5
gamma = 1/7
sigma = 1/5

beta_er = R0_target * gamma / q_er
beta_ba = R0_target * gamma / q_ba

# SEIR schema
SEIR_schema = (
    fg.ModelSchema("SEIR")
    .define_compartment(['S','E','I','R'])
    .add_network_layer('contact')
    .add_edge_interaction('infection',from_state='S',to_state='E',inducer='I',network_layer='contact',rate='beta')
    .add_node_transition('progress',from_state='E',to_state='I',rate='sigma')
    .add_node_transition('recover',from_state='I',to_state='R',rate='gamma')
)

cfg_er = (fg.ModelConfiguration(SEIR_schema)
          .add_parameter(beta=float(beta_er),sigma=sigma,gamma=gamma)
          .get_networks(contact=er_csr))
cfg_ba = (fg.ModelConfiguration(SEIR_schema)
          .add_parameter(beta=float(beta_ba),sigma=sigma,gamma=gamma)
          .get_networks(contact=ba_csr))

init_percent = {'S':99,'E':0,'I':1,'R':0}
init_cond = {'percentage': init_percent}
stop={'time':200}
nsim=20

sim_er = fg.Simulation(cfg_er, initial_condition=init_cond, stop_condition=stop, nsim=nsim)
sim_er.run()

time_er, state_er, *_ = sim_er.get_results()

er_df = pd.DataFrame({'time':time_er,'S':state_er[0],'E':state_er[1],'I':state_er[2],'R':state_er[3]})

er_df.to_csv(os.path.join(output_dir,'results-11.csv'),index=False)

sim_ba = fg.Simulation(cfg_ba, initial_condition=init_cond, stop_condition=stop, nsim=nsim)
sim_ba.run()

time_ba, state_ba, *_ = sim_ba.get_results()

ba_df = pd.DataFrame({'time':time_ba,'S':state_ba[0],'E':state_ba[1],'I':state_ba[2],'R':state_ba[3]})
ba_df.to_csv(os.path.join(output_dir,'results-12.csv'),index=False)

# Deterministic homogeneous ODE
beta_hom = R0_target * gamma

def seir_ode(t,y):
    S,E,I,R = y
    dS = -beta_hom*S*I
    dE = beta_hom*S*I - sigma*E
    dI = sigma*E - gamma*I
    dR = gamma*I
    return [dS,dE,dI,dR]

y0=[0.99,0,0.01,0]
sol=solve_ivp(seir_ode,[0,200],y0,t_eval=np.linspace(0,200,1001))

det_df = pd.DataFrame({'time':sol.t,'S':sol.y[0]*len(k_er),'E':sol.y[1]*len(k_er),'I':sol.y[2]*len(k_er),'R':sol.y[3]*len(k_er)})

det_df.to_csv(os.path.join(output_dir,'results-13.csv'),index=False)

import pandas as pd, os, numpy as np
out=os.path.join(os.getcwd(),'output')

er=pd.read_csv(os.path.join(out,'results-11.csv'))
ba=pd.read_csv(os.path.join(out,'results-12.csv'))
det=pd.read_csv(os.path.join(out,'results-13.csv'))

metrics={}
for name,df in [('ER',er),('BA',ba),('ODE',det)]:
    peak_I=df['I'].max(); peak_time=float(df.loc[df['I'].idxmax(),'time'])
    final_size=df['R'].iloc[-1]
    duration=df.loc[df['I']>1].shape[0]  # time steps with >1 infectious nodes
    metrics[name]={'peak_I':peak_I,'peak_time':peak_time,'final_R':final_size,'duration_steps':duration}
metrics
import pandas as pd, os
out=os.path.join(os.getcwd(),'output')

er=pd.read_csv(os.path.join(out,'results-11.csv'))
ba=pd.read_csv(os.path.join(out,'results-12.csv'))
det=pd.read_csv(os.path.join(out,'results-13.csv'))

summary={}
for name,df in [('ER',er),('BA',ba),('ODE',det)]:
    peak_I=df['I'].max(); peak_time=float(df.loc[df['I'].idxmax(),'time'])
    final_R=df['R'].iloc[-1]
    # first and last time with I>1
    above=df[df['I']>1]
    if not above.empty:
        duration=above['time'].iloc[-1]-above['time'].iloc[0]
    else:
        duration=0
    summary[name]={'peak_I':peak_I,'peak_time':peak_time,'final_R':final_R,'duration':duration}
summary
import networkx as nx, os, scipy.sparse as sparse, numpy as np, json, matplotlib.pyplot as plt
N=5000
mean_k=10
p=mean_k/(N-1)
G_er=nx.fast_gnp_random_graph(N,p,seed=42)
G_ba=nx.barabasi_albert_graph(N,5,seed=42)
# compute moments
def moments(G):
    degrees=np.array([d for _,d in G.degree()])
    k=degrees.mean()
    k2=(degrees**2).mean()
    return float(k), float(k2)
mk_er,k2_er=moments(G_er)
mk_ba,k2_ba=moments(G_ba)
# save networks
out_dir=os.path.join(os.getcwd(),'output')
os.makedirs(out_dir,exist_ok=True)
sparse.save_npz(os.path.join(out_dir,'network_er.npz'), nx.to_scipy_sparse_array(G_er))
sparse.save_npz(os.path.join(out_dir,'network_ba.npz'), nx.to_scipy_sparse_array(G_ba))
# degree distribution plot
plt.figure()
plt.hist([d for _,d in G_er.degree()],bins=40)
plt.xlabel('degree');plt.ylabel('count');plt.title('ER degree distribution')
plt.savefig(os.path.join(out_dir,'degree_er.png'))
plt.close()
plt.figure()
plt.hist([d for _,d in G_ba.degree()],bins=40)
plt.xlabel('degree');plt.ylabel('count');plt.title('BA degree distribution')
plt.savefig(os.path.join(out_dir,'degree_ba.png'))
plt.close()
json_out={'mk_er':mk_er,'k2_er':k2_er,'mk_ba':mk_ba,'k2_ba':k2_ba}
