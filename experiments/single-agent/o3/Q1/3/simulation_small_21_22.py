
import networkx as nx, numpy as np, os, scipy.sparse as sparse, random, math, pandas as pd, time
random.seed(42)
np.random.seed(42)
output_dir=os.path.join(os.getcwd(),'output'); os.makedirs(output_dir,exist_ok=True)
N=2000
k_avg=10
# regenerate networks smaller
p=k_avg/(N-1)
G_er_nx = nx.fast_gnp_random_graph(N,p,seed=1)
G_ba_nx = nx.barabasi_albert_graph(N,k_avg//2,seed=1)
# save
sparse.save_npz(os.path.join(output_dir,'network_er_small.npz'), nx.to_scipy_sparse_array(G_er_nx))
sparse.save_npz(os.path.join(output_dir,'network_ba_small.npz'), nx.to_scipy_sparse_array(G_ba_nx))
# compute metrics
k_er=np.array([d for _,d in G_er_nx.degree()])
mean_er=k_er.mean(); second_er=(k_er**2).mean(); q_er=(second_er-mean_er)/mean_er
k_ba=np.array([d for _,d in G_ba_nx.degree()])
mean_ba=k_ba.mean(); second_ba=(k_ba**2).mean(); q_ba=(second_ba-mean_ba)/mean_ba
sigma=1/3; gamma=1/5; R0=2.5
beta_er=R0*gamma/q_er; beta_ba=R0*gamma/q_ba
print('beta',beta_er,beta_ba)
# convert to csr for simulation
G_er=sparse.load_npz(os.path.join(output_dir,'network_er_small.npz'))
G_ba=sparse.load_npz(os.path.join(output_dir,'network_ba_small.npz'))
N=G_er.shape[0]
neighbors_er=[G_er.indices[G_er.indptr[i]:G_er.indptr[i+1]] for i in range(N)]
neighbors_ba=[G_ba.indices[G_ba.indptr[i]:G_ba.indptr[i+1]] for i in range(N)]

def simulate(neighbors,beta,runs=20,max_days=160):
    results=[]
    N=len(neighbors)
    for run in range(runs):
        state=np.zeros(N,dtype=np.int8)
        initial=np.random.choice(N,5,replace=False)
        state[initial]=2
        ts=[]
        for t in range(max_days):
            S=(state==0).sum();E=(state==1).sum();I=(state==2).sum();R=(state==3).sum();
            ts.append([t,S,E,I,R])
            if I==0 and E==0:
                break
            new=state.copy()
            # infection step
            susceptible_indices=np.where(state==0)[0]
            for node in susceptible_indices:
                inf_neighbors=(state[neighbors[node]]==2).sum()
                if inf_neighbors and random.random() < 1-math.exp(-beta*inf_neighbors):
                    new[node]=1
            # expos -> infectious
            e_idx=np.where(state==1)[0]
            for node in e_idx:
                if random.random() < sigma:
                    new[node]=2
            i_idx=np.where(state==2)[0]
            for node in i_idx:
                if random.random() < gamma:
                    new[node]=3
            state=new
        results.append(np.array(ts))
    return results
start=time.time()
res_er=simulate(neighbors_er,beta_er)
res_ba=simulate(neighbors_ba,beta_ba)
print('sim done',time.time()-start)
max_len=max(max(len(r) for r in res_er), max(len(r) for r in res_ba))
agg_er=np.zeros((max_len,5));
agg_ba=np.zeros((max_len,5));
for r in res_er:
    agg_er[:len(r)] += r
agg_er/=len(res_er)
for r in res_ba:
    agg_ba[:len(r)] += r
agg_ba/=len(res_ba)
import pandas as pd
df_er=pd.DataFrame(agg_er,columns=['time','S','E','I','R'])
df_ba=pd.DataFrame(agg_ba,columns=['time','S','E','I','R'])
csv_er=os.path.join(output_dir,'results-21.csv')
csv_ba=os.path.join(output_dir,'results-22.csv')
df_er.to_csv(csv_er,index=False)
df_ba.to_csv(csv_ba,index=False)
print({'csv_er':csv_er,'csv_ba':csv_ba})