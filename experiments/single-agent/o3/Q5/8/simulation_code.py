
import os, numpy as np, networkx as nx, scipy.sparse as sparse, fastgemf as fg, pandas as pd, random
np.random.seed(42)
random.seed(42)
N=10000
x=0.125 # fraction degree 10
n10=int(N*x)
deg_list=[10]*n10+[2]*(N-n10)
if sum(deg_list)%2==1:
    deg_list[-1]+=1
G=nx.configuration_model(deg_list,create_using=nx.Graph())
G.remove_edges_from(nx.selfloop_edges(G))
# save network
out_dir=os.path.join(os.getcwd(),'output')
os.makedirs(out_dir,exist_ok=True)
sparse.save_npz(os.path.join(out_dir,'network.npz'),nx.to_scipy_sparse_array(G))
# calc degree moments
k=np.array([d for _,d in G.degree()])
mean_k=k.mean(); mean_k2=(k**2).mean(); q=(mean_k2-mean_k)/mean_k
# parameters
R0=4
beta=R0/q # with gamma=1

gamma=1.0
# set up SIR schema
SIR_schema=(fg.ModelSchema('SIR').define_compartment(['S','I','R']).add_network_layer('contact').add_node_transition(name='recovery',from_state='I',to_state='R',rate='gamma').add_edge_interaction(name='infection',from_state='S',to_state='I',inducer='I',network_layer='contact',rate='beta'))
config=(fg.ModelConfiguration(SIR_schema).add_parameter(beta=beta,gamma=gamma).get_networks(contact=nx.to_scipy_sparse_array(G)))

# helper to run scenario
def run_scenario(vaccination_mask,scenario_id):
    # vaccination_mask bool array length N True if vaccinated (removed)
    susceptible_nodes=np.where(~vaccination_mask)[0]
    size=len(susceptible_nodes)
    # initial infection 1% among susceptible
    n_inf=max(1,int(0.01*size))
    infected_nodes=np.random.choice(susceptible_nodes,n_inf,replace=False)
    X0=np.zeros(N,dtype=int) # 0=>S
    X0[infected_nodes]=1 # I
    X0[vaccination_mask]=2 # Use R compartment for vaccinated immune
    initial={'exact':X0}
    sim=fg.Simulation(config,initial_condition=initial,stop_condition={'time':40},nsim=3)
    sim.run()
    time,counts,*_=sim.get_results()
    # counts shape C x T
    data={'time':time}
    for i,c in enumerate(SIR_schema.compartments):
        data[c]=counts[i,:]
    df=pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir,f'results-1{scenario_id}.csv'),index=False)
    sim.plot_results(show_figure=False,save_figure=True,save_path=os.path.join(out_dir,f'results-1{scenario_id}.png'))
    # derive metrics
    final_size=counts[2,-1]/N
    peak_I=counts[1,:].max()/N
    peak_time=time[counts[1,:].argmax()]
    return {'scenario':scenario_id,'vaccinated_prop':vaccination_mask.mean(),'final_size':final_size,'peak_I':peak_I,'peak_time':peak_time}

vacc_none=np.zeros(N,dtype=bool)
vacc_random=np.zeros(N,dtype=bool)
rand_vac_idx=np.random.choice(N,int(0.75*N),replace=False)
vacc_random[rand_vac_idx]=True
# targeted deg10
nodes_deg10=[n for n,d in G.degree() if d==10]
vacc_target=np.zeros(N,dtype=bool)
vacc_target[nodes_deg10]=True

metrics=[]
metrics.append(run_scenario(vacc_none,0))
metrics.append(run_scenario(vacc_random,1))
metrics.append(run_scenario(vacc_target,2))
metrics