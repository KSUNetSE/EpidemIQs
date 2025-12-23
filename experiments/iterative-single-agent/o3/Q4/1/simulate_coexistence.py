
import os, numpy as np, networkx as nx, scipy.sparse as sparse, pandas as pd
from numpy.random import default_rng
output_dir=os.path.join(os.getcwd(),'output')
G_A=sparse.load_npz(os.path.join(output_dir,'layerA.npz'))
G_B1=sparse.load_npz(os.path.join(output_dir,'layerB1.npz'))
G_B2=sparse.load_npz(os.path.join(output_dir,'layerB2.npz'))

N=G_A.shape[0]

def simulate(G1,G2,beta1,beta2,delta1=1.0,delta2=1.0,T=200,dt=0.1,init=10,seed=0):
    rng=default_rng(seed)
    neighbors1=[G1.indices[G1.indptr[i]:G1.indptr[i+1]] for i in range(N)]
    neighbors2=[G2.indices[G2.indptr[i]:G2.indptr[i+1]] for i in range(N)]
    state=np.zeros(N,dtype=np.int8)
    infected1=rng.choice(N,init,replace=False)
    infected2=rng.choice(list(set(range(N))-set(infected1)),init,replace=False)
    state[infected1]=1
    state[infected2]=2
    times=[];I1=[];I2=[]
    t=0
    while t<=T:
        times.append(t);I1.append((state==1).sum());I2.append((state==2).sum())
        new_state=state.copy()
        for node in range(N):
            if state[node]==0:
                # exposures
                if any(state[nb]==1 for nb in neighbors1[node]):
                    if rng.random()<1-np.exp(-beta1*dt):
                        new_state[node]=1
                        continue
                if any(state[nb]==2 for nb in neighbors2[node]):
                    if rng.random()<1-np.exp(-beta2*dt):
                        new_state[node]=2
            elif state[node]==1:
                if rng.random()<1-np.exp(-delta1*dt):
                    new_state[node]=0
            elif state[node]==2:
                if rng.random()<1-np.exp(-delta2*dt):
                    new_state[node]=0
        state=new_state
        t+=dt
    return np.array(times),np.array(I1),np.array(I2)

from numpy.linalg import eigvals
lambda1_A=eigvals(G_A.toarray()).real.max()
lambda1_B2=eigvals(G_B2.toarray()).real.max()

beta1=4.0/lambda1_A
beta2=4.0/lambda1_B2
print('tau1',beta1/lambda1_A,'tau2',beta2/lambda1_B2)

time1,I11,I12=simulate(G_A,G_B1,beta1,beta1,seed=1,T=300)
time2,I21,I22=simulate(G_A,G_B2,beta1,beta2,seed=2,T=300)
print(I11[-1],I12[-1],I21[-1],I22[-1])
