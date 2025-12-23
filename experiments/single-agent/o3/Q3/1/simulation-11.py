
import fastgemf as fg, os, numpy as np, scipy.sparse as sparse, random, pandas as pd
output_dir=os.path.join(os.getcwd(),'output')

# load networks
A=sparse.load_npz(os.path.join(output_dir,'networkA.npz'))
B=sparse.load_npz(os.path.join(output_dir,'networkB.npz'))

lam_A=14.42160761348478
lam_B=13.801791187060179

delta1=1.0
delta2=1.0
beta1=1.2/lam_A
beta2=1.2/lam_B

# Define SI1SI2S model
SI1SI2S_schema=(fg.ModelSchema('SI1SI2S')
    .define_compartment(['S','I1','I2'])
    .add_network_layer('layerA')
    .add_network_layer('layerB')
    .add_node_transition(name='recov1',from_state='I1',to_state='S',rate='delta1')
    .add_node_transition(name='recov2',from_state='I2',to_state='S',rate='delta2')
    .add_edge_interaction(name='inf1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
    .add_edge_interaction(name='inf2',from_state='S',to_state='I2',inducer='I2',network_layer='layerB',rate='beta2')
)

config=(fg.ModelConfiguration(SI1SI2S_schema)
    .add_parameter(beta1=beta1,beta2=beta2,delta1=delta1,delta2=delta2)
    .get_networks(layerA=A,layerB=B)
)
print(config)

# initial condition random 5% infected each
N=A.shape[0]
all_nodes=np.arange(N)
random.seed(2024)
I1_nodes=set(random.sample(list(all_nodes),int(0.05*N)))
remaining=list(set(all_nodes)-I1_nodes)
I2_nodes=set(random.sample(remaining,int(0.05*N)))
X0=np.zeros(N,dtype=int) # S=0,I1=1,I2=2 based on compartment order
for n in I1_nodes:
    X0[n]=1
for n in I2_nodes:
    X0[n]=2
initial={'exact':X0}

sim=fg.Simulation(config,initial_condition=initial,stop_condition={'time':200},nsim=1)

sim.run()

sim.plot_results(show_figure=False,save_figure=True,save_path=os.path.join(output_dir,'results-11.png'))

time,state_count,*_=sim.get_results()
res={'time':time}
for idx,comp in enumerate(SI1SI2S_schema.compartments):
    res[comp]=state_count[idx,:]
data=pd.DataFrame(res)
data.to_csv(os.path.join(output_dir,'results-11.csv'),index=False)

import fastgemf as fg, scipy.sparse as sparse, os, numpy as np, random, pandas as pd
out_dir=os.path.join(os.getcwd(),'output')
# Load networks
A=sparse.load_npz(os.path.join(out_dir,'layerA_network.npz'))
B=sparse.load_npz(os.path.join(out_dir,'layerB_network.npz'))
# Model schema
model_schema=(fg.ModelSchema('SI1SI2S')
              .define_compartment(['S','I1','I2'])
              .add_network_layer('layerA')
              .add_network_layer('layerB')
              .add_edge_interaction(name='infection1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
              .add_edge_interaction(name='infection2',from_state='S',to_state='I2',inducer='I2',network_layer='layerB',rate='beta2')
              .add_node_transition(name='recovery1',from_state='I1',to_state='S',rate='delta1')
              .add_node_transition(name='recovery2',from_state='I2',to_state='S',rate='delta2'))

# Parameters
beta1=0.104
beta2=0.244
delta1=1.0
delta2=1.0
model_instance=(fg.ModelConfiguration(model_schema)
                .add_parameter(beta1=beta1,beta2=beta2,delta1=delta1,delta2=delta2)
                .get_networks(layerA=A,layerB=B))
print(model_instance)
# Initial condition 5% each infection excluding overlap
N=A.shape[0]
num_I1=int(0.05*N)
num_I2=int(0.05*N)
all_nodes=list(range(N))
I1_nodes=random.sample(all_nodes,num_I1)
remaining=[n for n in all_nodes if n not in I1_nodes]
I2_nodes=random.sample(remaining,num_I2)
X0=np.zeros(N,dtype=int) # S is 0
X0[I1_nodes]=1
X0[I2_nodes]=2
initial_condition={'exact':X0}

sim=fg.Simulation(model_instance,initial_condition=initial_condition,stop_condition={'time':200},nsim=5)
sim.run()
sim.plot_results(show_figure=False,save_figure=True,save_path=os.path.join(out_dir,'results-11.png'))
time,state_cnt,*_=sim.get_results()
results={'time':time}
for idx,comp in enumerate(model_schema.compartments):
    results[comp]=state_cnt[idx,:]
DF=pd.DataFrame(results)
DF.to_csv(os.path.join(out_dir,'results-11.csv'),index=False)