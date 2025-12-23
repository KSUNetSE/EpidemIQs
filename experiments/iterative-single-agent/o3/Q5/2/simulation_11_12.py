
import os, numpy as np, scipy.sparse as sparse, fastgemf as fg, pandas as pd, random
from scipy import sparse
import networkx as nx

output_dir=os.path.join(os.getcwd(),'output')

# Load network
G_csr=sparse.load_npz(os.path.join(output_dir,'network.npz'))
N=G_csr.shape[0]

# Model schema SIR
SIR_schema=(fg.ModelSchema('SIR')
            .define_compartment(['S','I','R'])
            .add_network_layer('contact')
            .add_edge_interaction('infection',from_state='S',to_state='I',inducer='I',network_layer='contact',rate='beta')
            .add_node_transition('recovery',from_state='I',to_state='R',rate='gamma'))

beta=0.0923
gamma=0.1

sir_instance=(fg.ModelConfiguration(SIR_schema)
              .add_parameter(beta=beta,gamma=gamma)
              .get_networks(contact=G_csr))

# Function to simulate scenario and save
import itertools, math

def simulate(vaccination_nodes,filename_prefix):
    # vaccination_nodes: list of node indices to set as R initially
    # Choose 10 random susceptible nodes to infect among non-vaccinated
    susceptible_nodes=[i for i in range(N) if i not in vaccination_nodes]
    initial_infected=random.sample(susceptible_nodes,10)
    X0=np.zeros(N,dtype=int)
    X0[vaccination_nodes]=2 # R state index 2
    X0[initial_infected]=1 # I state index 1
    initial_condition={'exact':X0}
    sim=fg.Simulation(sir_instance,initial_condition=initial_condition,stop_condition={'time':300},nsim=5)
    sim.run()
    # save plot and csv
    png_path=os.path.join(output_dir,f'{filename_prefix}.png')
    csv_path=os.path.join(output_dir,f'{filename_prefix}.csv')
    sim.plot_results(show_figure=False,save_figure=True,save_path=png_path)
    time, state_counts, *_=sim.get_results()
    data={'time':time}
    comps=SIR_schema.compartments
    for idx,comp in enumerate(comps):
        data[comp]=state_counts[idx,:]
    pd.DataFrame(data).to_csv(csv_path,index=False)
    return png_path,csv_path

# Scenario 1: random vaccination 75%
random_vac_nodes=random.sample(range(N),int(0.75*N))
run1png, run1csv=simulate(random_vac_nodes,'results-11')

# Scenario 2: targeted vaccination all degree10 nodes
# Need to retrieve degree list from network; load earlier degree sequence
G_nx=nx.from_scipy_sparse_array(G_csr)
deg_seq=dict(G_nx.degree())
target_nodes=[n for n,d in deg_seq.items() if d==10]
run2png, run2csv=simulate(target_nodes,'results-12')

print(run1png,run1csv,run2png,run2csv)
