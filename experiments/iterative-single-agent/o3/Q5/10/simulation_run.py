
import os, scipy.sparse as sparse, numpy as np, fastgemf as fg, json, random
output_dir = os.path.join(os.getcwd(),'output')
net_path = os.path.join(output_dir,'network.npz')
G_csr = sparse.load_npz(net_path)
# read stats
a = open(os.path.join(output_dir,'network_stats.json'))
stats = json.load(a)
mean_k = stats['mean_k']
q = stats['q']

gamma = 1.0
beta = 4 / q  # R0=4
print(beta)

# define model schema
SIR_schema = (
    fg.ModelSchema('SIR')
    .define_compartment(['S','I','R'])
    .add_network_layer('contacts')
    .add_edge_interaction('infection',from_state='S',to_state='I',inducer='I',network_layer='contacts',rate='beta')
    .add_node_transition('recovery',from_state='I',to_state='R',rate='gamma')
)

SIR_instance = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contacts=G_csr)
)
print(SIR_instance)

N = G_csr.shape[0]
random.seed(0)
np.random.seed(0)

# function to create initial_condition dict

def ic_random_vaccination(vacc_fraction):
    X0 = np.zeros(N, dtype=int)  # S=0
    vaccinated = np.random.choice(N, size=int(vacc_fraction*N), replace=False)
    X0[vaccinated] = 2  # R
    # infect 1% of remaining susceptibles
    susceptible_indices = np.where(X0==0)[0]
    infected = np.random.choice(susceptible_indices, size=int(0.01*N), replace=False)
    X0[infected] = 1
    return {'exact': X0}

# targeted vaccination degree 10
import networkx as nx
G = nx.from_scipy_sparse_array(G_csr)

deg10_nodes = [n for n,d in G.degree() if d==10]
print('deg10 fraction', len(deg10_nodes)/N)

def ic_targeted(alpha=1.0):
    X0 = np.zeros(N, dtype=int)
    vacc_count = int(alpha*len(deg10_nodes))
    vacc_nodes = random.sample(deg10_nodes, vacc_count)
    X0[vacc_nodes] = 2
    # infect 1% random others
    susceptible_indices = np.where(X0==0)[0]
    infected = np.random.choice(susceptible_indices, size=int(0.01*N), replace=False)
    X0[infected] = 1
    return {'exact': X0}

# run simulations
sim1 = fg.Simulation(SIR_instance, initial_condition=ic_random_vaccination(0.75), stop_condition={'time':100}, nsim=3)
sim1.run()
sim1.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(output_dir,'results-11.png'))
# save csv
import pandas as pd

time, state_count, *_ = sim1.get_results()
res = {'time': time}
for idx,comp in enumerate(SIR_schema.compartments):
    res[comp] = state_count[idx,:]

df=pd.DataFrame(res)
df.to_csv(os.path.join(output_dir,'results-11.csv'),index=False)

# targeted vacc all deg10
sim2= fg.Simulation(SIR_instance, initial_condition=ic_targeted(), stop_condition={'time':100}, nsim=3)
sim2.run()
sim2.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(output_dir,'results-12.png'))
time2,state_count2,*_=sim2.get_results()
res2={'time':time2}
for idx,comp in enumerate(SIR_schema.compartments):
    res2[comp]=state_count2[idx,:]

df2=pd.DataFrame(res2)
df2.to_csv(os.path.join(output_dir,'results-12.csv'),index=False)
