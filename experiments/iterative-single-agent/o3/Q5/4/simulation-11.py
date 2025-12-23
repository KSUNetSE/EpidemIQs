
import os, networkx as nx, scipy.sparse as sp, random, pandas as pd, numpy as np, matplotlib.pyplot as plt, collections, math, time
random.seed(42)
output_dir=os.path.join(os.getcwd(),'output')
csr=sp.load_npz(os.path.join(output_dir,'network.npz'))
G=nx.from_scipy_sparse_array(csr)
N=G.number_of_nodes()


def simulate_SI(G, max_steps=100):
    # Infect one random node, discrete time steps, p=1 for infection
    infected=set([random.choice(list(G.nodes()))])
    times=[0]
    sizes=[1]
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
        if len(sizes[-1:]) and sizes[-1]==N: # all infected
            break
    return times, sizes

# scenario A: random vacc 75%
frac=0.75
rand_nodes=random.sample(list(G.nodes()), int(frac*N))
G_rand=G.copy()
G_rand.remove_nodes_from(rand_nodes)

# run 3 simulations save aggregated mean series length maybe variable; we will run one for demonstration

times, sizes=simulate_SI(G_rand)

# Save to csv
csv_path=os.path.join(output_dir,'results-11.csv')
pd.DataFrame({'time':times,'cum_infected':sizes}).to_csv(csv_path,index=False)

# Plot
plt.figure()
plt.plot(times,sizes)
plt.xlabel('time step')
plt.ylabel('cumulative infected')
plt.title('Random vaccination 75%')
plt.tight_layout()
plt_path=os.path.join(output_dir,'results-11.png')
plt.savefig(plt_path)
plt.close()

# scenario B: vaccinate k=10 nodes
k10_nodes=[n for n,d in G.degree() if d==10]
G_tar=G.copy()
G_tar.remove_nodes_from(k10_nodes)
Btimes,Bsizes=simulate_SI(G_tar)

csv_path2=os.path.join(output_dir,'results-12.csv')
pd.DataFrame({'time':Btimes,'cum_infected':Bsizes}).to_csv(csv_path2,index=False)

plt.figure()
plt.plot(Btimes,Bsizes)
plt.xlabel('time step')
plt.ylabel('cumulative infected')
plt.title('Vaccinate degree 10 nodes')
plt.tight_layout()
plt_path2=os.path.join(output_dir,'results-12.png')
plt.savefig(plt_path2)
plt.close()
return_dict={'rand_len':len(G_rand),'tar_len':len(G_tar)}
