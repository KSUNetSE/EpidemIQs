
import os, networkx as nx, random, scipy.sparse as sp, pandas as pd
output_dir=os.path.join(os.getcwd(),'output')
csr=sp.load_npz(os.path.join(output_dir,'network.npz'))
G=nx.from_scipy_sparse_array(csr)

# baseline SI simulation (same as earlier function) but produce csv baseline
import numpy as np
random.seed(0)

def simulate_SI(G, max_steps=50):
    infected=set(random.sample(list(G.nodes()),10))
    times=[0]
    sizes=[len(infected)]
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
    return times, sizes

(times0,sizes0)=simulate_SI(G)

pd.DataFrame({'time':times0,'cum_infected':sizes0}).to_csv(os.path.join(output_dir,'results-10.csv'),index=False)

