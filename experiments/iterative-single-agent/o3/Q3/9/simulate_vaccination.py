
import networkx as nx, random, numpy as np, os, scipy.sparse as sp
random.seed(1)
np.random.seed(1)
csr=sp.load_npz(os.path.join(os.getcwd(),'output','network.npz'))
G=nx.from_scipy_sparse_array(csr)
N=G.number_of_nodes()


def run_sir(G, beta, gamma, vaccinated_nodes, init_infected=10, max_steps=200):
    # 0=S,1=I,2=R/V
    state=np.zeros(N, dtype=int)
    for v in vaccinated_nodes:
        state[v]=2
    susceptible_nodes=[i for i in range(N) if state[i]==0]
    infected_initial=random.sample(susceptible_nodes, init_infected)
    for i in infected_initial:
        state[i]=1
    S_counts=[]; I_counts=[]; R_counts=[]
    for t in range(max_steps):
        S_counts.append((state==0).sum())
        I_counts.append((state==1).sum())
        R_counts.append((state==2).sum())
        if I_counts[-1]==0:
            break
        new_infections=[]
        recoveries=[]
        for i in np.where(state==1)[0]:
            for neigh in G[i]:
                if state[neigh]==0 and random.random()<beta:
                    new_infections.append(neigh)
            if random.random()<gamma:
                recoveries.append(i)
        for n in new_infections:
            state[n]=1
        for r in recoveries:
            state[r]=2
    return {'S':S_counts,'I':I_counts,'R':R_counts}

# parameters choose beta=0.25,gamma=0.0625? Actually beta/gamma=1. we pick beta=0.2,gamma=0.2 to maintain ratio 1
beta=0.2
gamma=0.2

# scenario 1 random vacc 75%
frac=0.75
vacc_nodes_random=set(random.sample(range(N), int(frac*N)))
res_random=run_sir(G,beta,gamma,vacc_nodes_random)

# scenario 2 vaccinate degree10 nodes
vacc_nodes_deg10={node for node,deg in G.degree() if deg==10}
res_deg10=run_sir(G,beta,gamma,vacc_nodes_deg10)

print('Counts lengths', len(res_random['I']), len(res_deg10['I']))
print('Final infections random',sum(res_random['R'])-len(vacc_nodes_random))
print('Final infections deg10',sum(res_deg10['R'])-len(vacc_nodes_deg10))