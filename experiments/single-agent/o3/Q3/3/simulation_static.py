
import os, numpy as np, scipy.sparse as sp, math, pandas as pd, matplotlib.pyplot as plt
N=10000
network_path=os.path.join(os.getcwd(),'output','network_static.npz')
A_csr=sp.load_npz(network_path)
deg=np.array(A_csr.sum(axis=1)).flatten()
mean_k=deg.mean()
mean_k2=(deg**2).mean()
q=(mean_k2-mean_k)/mean_k
gamma=0.2
beta=3*gamma/q
p_inf=1-math.exp(-beta)
print('mean_k',mean_k,'mean_k2',mean_k2,'q',q,'beta',beta,'p_inf',p_inf)

# Precompute neighbors list
indptr=A_csr.indptr
indices=A_csr.indices
neighbors=[indices[indptr[i]:indptr[i+1]] for i in range(N)]
T_max=200
sims=10
initial_infected=100
S=0;I=1;R=2

time_series_S=np.zeros(T_max+1)
time_series_I=np.zeros(T_max+1)
time_series_R=np.zeros(T_max+1)

for sim in range(sims):
    states=np.zeros(N,dtype=int)
    infected=np.random.choice(N,initial_infected,replace=False)
    states[infected]=I
    time_series_S[0]+=N-initial_infected
    time_series_I[0]+=initial_infected
    for t in range(1,T_max+1):
        current_I=np.where(states==I)[0]
        new_inf=[]
        for u in current_I:
            for v in neighbors[u]:
                if states[v]==S and np.random.random()<p_inf:
                    new_inf.append(v)
        states[new_inf]=I
        recovering=np.where((states==I)&(np.random.random(N)<(1-math.exp(-gamma))))[0]
        states[recovering]=R
        time_series_S[t]+=np.sum(states==S)
        time_series_I[t]+=np.sum(states==I)
        time_series_R[t]+=np.sum(states==R)
        if len(current_I)==0 and len(new_inf)==0:
            for tt in range(t+1,T_max+1):
                time_series_S[tt]+=np.sum(states==S)
                time_series_I[tt]+=0
                time_series_R[tt]+=np.sum(states==R)
            break

time_series_S/=sims
time_series_I/=sims
time_series_R/=sims

data=pd.DataFrame({'time':np.arange(T_max+1),'S':time_series_S,'I':time_series_I,'R':time_series_R})
static_csv=os.path.join(os.getcwd(),'output','results-13.csv')
data.to_csv(static_csv,index=False)
plt.figure()
plt.plot(data['time'],data['S'],label='S')
plt.plot(data['time'],data['I'],label='I')
plt.plot(data['time'],data['R'],label='R')
plt.title('Static BA Network Epidemic (avg 10 runs)')
plt.xlabel('Time')
plt.ylabel('Individuals')
plt.legend()
static_plot=os.path.join(os.getcwd(),'output','results-13.png')
plt.savefig(static_plot)
plt.close()
static_csv, static_plot