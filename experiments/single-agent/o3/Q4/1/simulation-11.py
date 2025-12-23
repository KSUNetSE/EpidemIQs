
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

import os, sys, numpy as np, scipy.sparse as sparse, networkx as nx, matplotlib.pyplot as plt
from numpy.random import default_rng

def run_simulation(G_A_csr, G_B_csr, beta1, delta1, beta2, delta2, initial_infected1_pct=1, initial_infected2_pct=1, t_max=300, nsim=5, rng_seed=42):
    rng = default_rng(rng_seed)
    N = G_A_csr.shape[0]
    # Precompute neighbor lists for efficiency
    neighbors_A = [G_A_csr.indices[G_A_csr.indptr[i]:G_A_csr.indptr[i+1]] for i in range(N)]
    neighbors_B = [G_B_csr.indices[G_B_csr.indptr[i]:G_B_csr.indptr[i+1]] for i in range(N)]

    results_time = None
    results_S = None
    results_I1 = None
    results_I2 = None

    for sim in range(nsim):
        # 0=Susceptible,1=I1,2=I2
        state = np.zeros(N, dtype=int)
        # infect initial percentages randomly
        idx = rng.permutation(N)
        n1 = int(initial_infected1_pct/100*N)
        n2 = int(initial_infected2_pct/100*N)
        state[idx[:n1]] = 1
        state[idx[n1:n1+n2]] = 2

        t=0.0
        dt=0.1
        time_series=[0]
        S_series=[(state==0).sum()]
        I1_series=[(state==1).sum()]
        I2_series=[(state==2).sum()]
        while t<t_max:
            # For each susceptible node compute infection probability from each meme this step
            # Use mean-field approximation dt small: prob adopt meme1 = 1-exp(-beta1*I_neighbors_A*dt) approx beta1*I_neighbors_A*dt (if small)
            new_state = state.copy()
            for node in range(N):
                if state[node]==0: # susceptible
                    infected1 = False
                    infected2 = False
                    # meme1 exposure
                    for nb in neighbors_A[node]:
                        if state[nb]==1:
                            if rng.random()<beta1*dt:
                                infected1=True
                                break
                    # meme2 exposure only if not infected1 already (exclusive)
                    if not infected1:
                        for nb in neighbors_B[node]:
                            if state[nb]==2:
                                if rng.random()<beta2*dt:
                                    infected2=True
                                    break
                    if infected1:
                        new_state[node]=1
                    elif infected2:
                        new_state[node]=2
                elif state[node]==1:
                    if rng.random()<delta1*dt:
                        new_state[node]=0
                elif state[node]==2:
                    if rng.random()<delta2*dt:
                        new_state[node]=0
            state = new_state
            t+=dt
            if int(t/dt)%10==0:
                time_series.append(t)
                S_series.append((state==0).sum())
                I1_series.append((state==1).sum())
                I2_series.append((state==2).sum())
        # aggregate
        time_arr = np.array(time_series)
        S_arr = np.array(S_series)
        I1_arr = np.array(I1_series)
        I2_arr = np.array(I2_series)
        if results_time is None:
            results_time=time_arr
            results_S=S_arr
            results_I1=I1_arr
            results_I2=I2_arr
        else:
            results_S+=S_arr
            results_I1+=I1_arr
            results_I2+=I2_arr
    # average
    results_S=results_S/nsim
    results_I1=results_I1/nsim
    results_I2=results_I2/nsim
    return results_time, results_S, results_I1, results_I2

output_dir=os.path.join(os.getcwd(),'output')
G_A=sparse.load_npz(os.path.join(output_dir,'layerA.npz'))
G_B1=sparse.load_npz(os.path.join(output_dir,'layerB1.npz'))
G_B2=sparse.load_npz(os.path.join(output_dir,'layerB2.npz'))

lambda1_A=np.linalg.eigvals(G_A.toarray()).real.max()
lambda1_B1=lambda1_A
lambda1_B2=np.linalg.eigvals(G_B2.toarray()).real.max()

# parameters
beta_factor=1.2
beta1=beta_factor/lambda1_A
beta2_identical=beta_factor/lambda1_B1
beta2_uncorr=beta_factor/lambda1_B2

delta1=1.0
delta2=1.0

# run for identical layers case

time1,S1,I1_1,I2_1=run_simulation(G_A,G_B1,beta1,delta1,beta2_identical,delta2)

# run for distinct layers case

time2,S2,I1_2,I2_2=run_simulation(G_A,G_B2,beta1,delta1,beta2_uncorr,delta2)

import pandas as pd

data1=pd.DataFrame({'time':time1,'S':S1,'I1':I1_1,'I2':I2_1})
data2=pd.DataFrame({'time':time2,'S':S2,'I1':I1_2,'I2':I2_2})

# save
results1_path=os.path.join(output_dir,'results-11.csv')
results2_path=os.path.join(output_dir,'results-12.csv')

data1.to_csv(results1_path,index=False)
data2.to_csv(results2_path,index=False)

# plot
plt.figure()
plt.plot(time1,I1_1,label='I1 (identical)')
plt.plot(time1,I2_1,label='I2 (identical)')
plt.xlabel('time')
plt.ylabel('infected')
plt.legend()
plt.savefig(os.path.join(output_dir,'results-11.png'))
plt.close()

plt.figure()
plt.plot(time2,I1_2,label='I1 (uncorr)')
plt.plot(time2,I2_2,label='I2 (uncorr)')
plt.xlabel('time')
plt.ylabel('infected')
plt.legend()
plt.savefig(os.path.join(output_dir,'results-12.png'))
plt.close()

