
import pandas as pd, numpy as np, os, scipy.integrate as spi
current_dir=os.getcwd(); output_dir=os.path.join(current_dir,'output')
N=2000
beta=0.35714285714285715
sigma=1/3
gamma=1/7
init={'S':98,'E':1,'I':1,'R':0}
S0=init['S']/100*N; E0=init['E']/100*N; I0=init['I']/100*N; R0=0

def seir(y,t):
    S,E,I,R=y
    dS=-beta*S*I/N
    dE=beta*S*I/N - sigma*E
    dI=sigma*E - gamma*I
    dR=gamma*I
    return [dS,dE,dI,dR]

time=np.arange(0,181)
ret=spi.odeint(seir,[S0,E0,I0,R0],time)

df=pd.DataFrame(ret,columns=['S','E','I','R'])
df['time']=time
hm_csv=os.path.join(output_dir,'results-11.csv'); df.to_csv(hm_csv,index=False)
peak_I=int(df['I'].max()); peak_time=int(df['time'][df['I'].idxmax()]); final_size=int(df['R'].iloc[-1])
summary={'peak_I_hom':peak_I,'peak_time_hom':peak_time,'final_size_hom':final_size}

# simulation-13.py BA network heterogeneous
import os, numpy as np, pandas as pd, scipy.sparse as sparse, fastgemf as fg, matplotlib.pyplot as plt
os.makedirs(os.path.join(os.getcwd(),'output'), exist_ok=True)
G_ba = sparse.load_npz(os.path.join(os.getcwd(),'output','network_ba.npz'))
mean_ba = 9.99
mean2_ba = 272.5716
R0_target=2.5
sigma=1/3
gamma=1/4
beta_ba = R0_target*gamma*mean_ba/(mean2_ba - mean_ba)
# define schema
schema = (
    fg.ModelSchema('SEIR')
    .define_compartment(['S','E','I','R'])
    .add_network_layer('contact')
    .add_edge_interaction('infection','S','E','I','contact','beta')
    .add_node_transition('progress','E','I','sigma')
    .add_node_transition('recovery','I','R','gamma')
)
config = (
    fg.ModelConfiguration(schema)
    .add_parameter(beta=beta_ba, sigma=sigma, gamma=gamma)
    .get_networks(contact=G_ba)
)
# initial condition same
ic={'percentage':{'E':1,'S':99}}
# simulation
sim=fg.Simulation(config,initial_condition=ic,stop_condition={'time':160},nsim=50)
sim.run()
# results
if sim.sim_cfg['nsim']>1:
    time,state_mean,state_var=sim.get_results()
else:
    time,state_mean=sim.get_results()
# save
comp=['S','E','I','R']
res={'time':time}
for idx,c in enumerate(comp):
    res[c]=state_mean[idx]

pd.DataFrame(res).to_csv(os.path.join(os.getcwd(),'output','results-13.csv'),index=False)
# plot
plt.figure(figsize=(6,4))
plt.plot(time,state_mean[2],label='Infected')
plt.plot(time,state_mean[1],label='Exposed')
plt.plot(time,state_mean[3],label='Recovered')
plt.xlabel('Time (days)');plt.ylabel('Population');
plt.title('SEIR on BA scale-free network (mean of 50 runs)')
plt.legend();plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'output','results-13.png'))
