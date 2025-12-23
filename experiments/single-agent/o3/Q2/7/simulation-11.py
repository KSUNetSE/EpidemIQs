
import fastgemf as fg, scipy.sparse as sp, numpy as np, os, json, networkx as nx, random
output_dir=os.path.join(os.getcwd(),'output')
G_csr=sp.load_npz(os.path.join(output_dir,'network.npz'))
# model schema
SIR_schema=(fg.ModelSchema('SIR')
            .define_compartment(['S','I','R'])
            .add_network_layer('contact')
            .add_node_transition(name='recovery',from_state='I',to_state='R',rate='gamma')
            .add_edge_interaction(name='infection',from_state='S',to_state='I',inducer='I',network_layer='contact',rate='beta'))

degrees=np.array(G_csr.sum(axis=1)).flatten()
N=len(degrees)
q=(np.mean(degrees**2)-np.mean(degrees))/np.mean(degrees)

def run_sim(beta,gamma,idx):
    config=(fg.ModelConfiguration(SIR_schema)
            .add_parameter(beta=beta,gamma=gamma)
            .get_networks(contact=G_csr))
    initial={'hubs_number':{'I':10,'S':N-10}}
    sim=fg.Simulation(config,initial_condition=initial,stop_condition={'time':160},nsim=20)
    sim.run()
    # save plot
    plot_path=os.path.join(output_dir,f'results-1{idx}.png')
    sim.plot_results(show_figure=False,save_figure=True,save_path=plot_path)
    time, state_count, *_ = sim.get_results()
    comp=['S','I','R']
    data={ 'time': time}
    for i,c in enumerate(comp):
        data[c]=state_count[i,:]
    import pandas as pd
    df=pd.DataFrame(data)
    csv_path=os.path.join(output_dir,f'results-1{idx}.csv')
    df.to_csv(csv_path,index=False)
    return csv_path,plot_path

gamma=1/7  # per day
# scenario 1: R0=0.8
R0=0.8
beta=R0*gamma/q
path1,plot1=run_sim(beta,gamma,1)
# scenario 2: R0=2.5
R0=2.5
beta=R0*gamma/q
path2,plot2=run_sim(beta,gamma,2)
print(path1,plot1,path2,plot2)
import fastgemf as fg, numpy as np, scipy.sparse as sparse, os, pandas as pd
G_csr = sparse.load_npz(os.path.join(os.getcwd(),'output','network.npz'))
# Model schema
SIR_schema=(fg.ModelSchema('SIR').define_compartment(['S','I','R']).add_network_layer('contact').add_edge_interaction('infection',from_state='S',to_state='I',inducer='I',network_layer='contact',rate='beta').add_node_transition('recovery',from_state='I',to_state='R',rate='gamma'))
# Parameters
beta_low=0.01767385913116828
beta_high=0.0441846478279207
gamma=0.143
scenario_params=[('low',beta_low),('high',beta_high)]
results_paths=[]
for idx,(label,beta) in enumerate(scenario_params):
    config=(fg.ModelConfiguration(SIR_schema).add_parameter(beta=beta,gamma=gamma).get_networks(contact=G_csr))
    # initial condition: 5 infected
    initial={'percentage':{'I':1,'R':0,'S':99}}
    sim=fg.Simulation(config,initial_condition=initial,stop_condition={'time':365},nsim=5)
    sim.run()
    time,state_count,*_=sim.get_results()
    data={'time':time}
    for i,comp in enumerate(SIR_schema.compartments):
        data[comp]=state_count[i,:]
    df=pd.DataFrame(data)
    csv_path=os.path.join(os.getcwd(),'output',f'results-1{idx+1}.csv')
    df.to_csv(csv_path,index=False)
    sim.plot_results(show_figure=False,save_figure=True,save_path=os.path.join(os.getcwd(),'output',f'results-1{idx+1}.png'))
    results_paths.append((csv_path,f'output/results-1{idx+1}.png'))
print(results_paths)